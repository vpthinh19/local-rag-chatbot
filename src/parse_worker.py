"""Disposable CLI worker for LiteParse ingestion and Markdown chunking."""

import argparse
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Protocol

from src.config import settings
from src.models import Chunk, DataValidationError


MAX_CHUNK_TOKENS = 1_024
_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True, slots=True)
class PageMarkdown:
    """Markdown extracted from one physical document page."""

    page_num: int
    markdown: str


@dataclass(frozen=True, slots=True)
class _PageSpan:
    """Character interval mapping joined Markdown back to a page."""

    start: int
    end: int
    page_num: int


class _TokenCounter(Protocol):
    """Minimal tokenizer contract needed to enforce chunk size."""

    def __call__(
        self, text: str, *, add_special_tokens: bool = False
    ) -> int:
        """Count model tokens in text."""
        ...


def build_chunks(
    pages: Sequence[PageMarkdown],
    *,
    file_id: str,
    file_name: str,
    split_indices: Callable[[str], Sequence[tuple[int, str]]],
    token_count: _TokenCounter,
) -> list[Chunk]:
    """Split joined page Markdown and attach source-page references."""
    document, spans = _join_pages(pages)
    if not document.strip() or not spans:
        raise DataValidationError("parsed document is empty")

    chunks: list[Chunk] = []
    for chunk_id, item in enumerate(split_indices(document)):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or not isinstance(item[1], str)
        ):
            raise DataValidationError("splitter returned an invalid chunk index")
        offset, text = item
        if not text.strip():
            raise DataValidationError("splitter returned an empty chunk")
        end = offset + len(text)
        if offset < 0 or end > len(document) or document[offset:end] != text:
            raise DataValidationError("splitter returned an invalid character span")
        if token_count(text, add_special_tokens=False) > MAX_CHUNK_TOKENS:
            raise DataValidationError("chunk exceeds the 1024-token payload limit")

        # Splitter offsets let citations survive whole-document chunking.
        first_page = _page_at(spans, offset)
        last_page = _page_at(spans, end - 1)
        reference = (
            f"p. {first_page}"
            if first_page == last_page
            else f"pp. {first_page}-{last_page}"
        )
        chunks.append(
            Chunk(file_id, file_name, chunk_id, [reference], text)
        )

    if not chunks:
        raise DataValidationError("splitter produced no usable chunks")
    return chunks


def parse_file(
    input_path: Path,
    file_id: str,
    file_name: str,
    max_pages: int,
    tokenizer_name: str,
) -> list[Chunk]:
    """Parse, chunk, and release all heavyweight worker-local objects."""
    parser = result = tokenizer = splitter = pages = None
    try:
        from liteparse import LiteParse

        parser = LiteParse(
            ocr_enabled=True,
            ocr_language="vie+eng",
            max_pages=max_pages,
            dpi=150,
            output_format="markdown",
            preserve_very_small_text=False,
            image_mode="placeholder",
            extract_links=True,
        )
        result = parser.parse(str(input_path))
        pages = [
            PageMarkdown(
                page_num=int(page.page_num),
                markdown=str(page.markdown or ""),
            )
            for page in result.pages
        ]
        tokenizer, splitter = _load_splitter(tokenizer_name)

        def count_tokens(
            text: str, *, add_special_tokens: bool = False
        ) -> int:
            """Adapt the tokenizer encoding API to a token count."""
            return len(
                tokenizer.encode(
                    text, add_special_tokens=add_special_tokens
                ).ids
            )

        return build_chunks(
            pages,
            file_id=file_id,
            file_name=file_name,
            split_indices=splitter.chunk_indices,
            token_count=count_tokens,
        )
    finally:
        # Drop heavyweight references before the worker serializes output and exits.
        parser = result = tokenizer = splitter = pages = None


def _load_splitter(tokenizer_name: str):
    """Load the local tokenizer and token-bounded Markdown splitter."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from semantic_text_splitter import MarkdownSplitter
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    splitter = MarkdownSplitter.from_huggingface_tokenizer(
        tokenizer, MAX_CHUNK_TOKENS, overlap=0
    )
    return tokenizer, splitter


def _join_pages(
    pages: Sequence[PageMarkdown],
) -> tuple[str, list[_PageSpan]]:
    """Join nonempty pages while recording their character intervals."""
    parts: list[str] = []
    spans: list[_PageSpan] = []
    cursor = 0
    previous_page = 0
    for page in pages:
        if (
            isinstance(page.page_num, bool)
            or not isinstance(page.page_num, int)
            or page.page_num <= previous_page
        ):
            raise DataValidationError("page numbers must be positive and increasing")
        previous_page = page.page_num
        markdown = page.markdown.strip()
        if not markdown:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(markdown)
        cursor += len(markdown)
        spans.append(_PageSpan(start, cursor, page.page_num))
    return "".join(parts), spans


def _page_at(spans: Sequence[_PageSpan], offset: int) -> int:
    """Resolve a joined-document character offset to its source page."""
    starts = [span.start for span in spans]
    index = bisect_right(starts, offset) - 1
    if index < 0:
        raise DataValidationError("chunk begins before the first page")
    return spans[index].page_num


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    """Atomically publish validated worker output for the parent process."""
    serialized = json.dumps(
        {"chunks": [chunk.to_dict() for chunk in chunks]},
        ensure_ascii=False,
        indent=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_arguments(
    input_path: Path, output_path: Path, file_id: str, file_name: str
) -> None:
    """Reject unsafe paths, identifiers, and unsupported file types."""
    if not input_path.is_file():
        raise DataValidationError("input file does not exist")
    if input_path.resolve() == output_path.resolve():
        raise DataValidationError("input and output paths must differ")
    if not _FILE_ID.fullmatch(file_id):
        raise DataValidationError("file ID is invalid")
    if (
        not file_name
        or Path(file_name).name != file_name
        or "/" in file_name
        or "\\" in file_name
        or Path(file_name).suffix.lower() not in {".pdf", ".docx"}
    ):
        raise DataValidationError("display filename is unsafe or unsupported")


def _parser() -> argparse.ArgumentParser:
    """Build the disposable worker's command-line interface."""
    parser = argparse.ArgumentParser(description="Parse one staged document")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--file-name", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    parse_document: Callable[
        [Path, str, str, int, str], list[Chunk]
    ] = parse_file,
) -> int:
    """Run one parse job and report failures through exit status and stderr."""
    args = _parser().parse_args(argv)
    try:
        _validate_arguments(args.input, args.output, args.file_id, args.file_name)
        chunks = parse_document(
            args.input,
            args.file_id,
            args.file_name,
            settings.max_parse_pages,
            settings.tokenizer_name,
        )
        if not isinstance(chunks, list) or not chunks or not all(
            isinstance(chunk, Chunk) for chunk in chunks
        ):
            raise DataValidationError("parser returned no validated chunks")
        _write_chunks(args.output, chunks)
        return 0
    except Exception as exc:
        message = " ".join(str(exc).split())[:500] or exc.__class__.__name__
        print(f"parse worker failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
