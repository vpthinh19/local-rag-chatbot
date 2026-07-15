import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import src.parse_worker as worker
from src.models import Chunk, DataValidationError
from src.parse_worker import PageMarkdown, build_chunks


def _token_count(text: str, *, add_special_tokens: bool = False) -> int:
    del add_special_tokens
    return len(text.split())


def test_multi_page_offsets_map_to_compact_page_refs() -> None:
    pages = [
        PageMarkdown(1, "# Mở đầu\nNội dung tiếng Việt"),
        PageMarkdown(2, "# Phần hai\nKết luận"),
    ]
    document = "# Mở đầu\nNội dung tiếng Việt\n\n# Phần hai\nKết luận"
    first = "# Mở đầu\nNội dung tiếng Việt"
    crossing = "Việt\n\n# Phần hai"
    split = lambda _: [
        (0, first),
        (document.index("Việt"), crossing),
        (document.index("Kết luận"), "Kết luận"),
    ]

    chunks = build_chunks(
        pages,
        file_id="file-1",
        file_name="tài-liệu.pdf",
        split_indices=split,
        token_count=_token_count,
    )

    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 2]
    assert [chunk.refs for chunk in chunks] == [["p. 1"], ["pp. 1-2"], ["p. 2"]]
    assert chunks[0].text == first
    assert chunks[1].file_name == "tài-liệu.pdf"


def test_empty_pages_or_chunks_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        build_chunks(
            [PageMarkdown(1, "  ")],
            file_id="f",
            file_name="a.pdf",
            split_indices=lambda _: [],
            token_count=_token_count,
        )

    with pytest.raises(DataValidationError, match="empty chunk"):
        build_chunks(
            [PageMarkdown(1, "text")],
            file_id="f",
            file_name="a.pdf",
            split_indices=lambda _: [(0, "  ")],
            token_count=_token_count,
        )


def test_chunk_token_bound_is_checked_without_special_tokens() -> None:
    calls: list[tuple[str, bool]] = []

    def token_count(text: str, *, add_special_tokens: bool = False) -> int:
        calls.append((text, add_special_tokens))
        return 1_025

    with pytest.raises(DataValidationError, match="1024"):
        build_chunks(
            [PageMarkdown(3, "too large")],
            file_id="f",
            file_name="a.pdf",
            split_indices=lambda _: [(0, "too large")],
            token_count=token_count,
        )

    assert calls == [("too large", False)]


def test_production_splitter_uses_configured_bge_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    tokenizer = object()
    splitter = object()

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(name: str):
            captured["name"] = name
            return tokenizer

    class FakeSplitter:
        @staticmethod
        def from_huggingface_tokenizer(value: object, capacity: int, overlap: int):
            captured.update(value=value, capacity=capacity, overlap=overlap)
            return splitter

    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer)
    )
    monkeypatch.setitem(
        sys.modules,
        "semantic_text_splitter",
        SimpleNamespace(MarkdownSplitter=FakeSplitter),
    )

    assert worker._load_splitter("BAAI/bge-m3") == (tokenizer, splitter)
    assert captured == {
        "name": "BAAI/bge-m3",
        "value": tokenizer,
        "capacity": 1024,
        "overlap": 0,
    }


def test_cli_writes_validated_plain_chunks_atomically(tmp_path: Path) -> None:
    input_path = tmp_path / "input.pdf"
    output_path = tmp_path / "chunks.json"
    input_path.write_bytes(b"fixture")
    expected = Chunk("file-1", "safe.pdf", 0, ["p. 1"], "Nội dung")

    def fake_parse(
        path: Path, file_id: str, file_name: str, max_pages: int, tokenizer: str
    ) -> list[Chunk]:
        assert path == input_path
        assert (file_id, file_name) == ("file-1", "safe.pdf")
        assert max_pages > 0
        assert tokenizer == "BAAI/bge-m3"
        return [expected]

    exit_code = worker.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--file-id",
            "file-1",
            "--file-name",
            "safe.pdf",
        ],
        parse_document=fake_parse,
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "chunks": [expected.to_dict()]
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_failure_is_bounded_and_does_not_write_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.pdf"
    output_path = tmp_path / "chunks.json"
    input_path.write_bytes(b"fixture")

    def fail(*args: object, **kwargs: object) -> list[Chunk]:
        del args, kwargs
        raise RuntimeError("sensitive " + "x" * 2_000)

    exit_code = worker.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--file-id",
            "file-1",
            "--file-name",
            "safe.pdf",
        ],
        parse_document=fail,
    )

    error = capsys.readouterr().err
    assert exit_code == 1
    assert "parse worker failed" in error
    assert len(error) < 600
    assert not output_path.exists()


@pytest.mark.parametrize("file_name", ["../escape.pdf", "bad.txt", ""])
def test_cli_rejects_unsafe_or_unsupported_names(
    tmp_path: Path, file_name: str
) -> None:
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"fixture")

    assert (
        worker.main(
            [
                "--input",
                str(input_path),
                "--output",
                str(tmp_path / "out.json"),
                "--file-id",
                "f",
                "--file-name",
                file_name,
            ],
            parse_document=lambda *args: [],
        )
        == 1
    )


def test_atomic_replace_failure_preserves_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "chunks.json"
    output.write_text('{"old":true}\n', encoding="utf-8")
    before = output.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(worker.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        worker._write_chunks(output, [Chunk("f", "a.pdf", 0, [], "text")])

    assert output.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_fastapi_side_modules_do_not_import_parser_stack() -> None:
    banned = ("liteparse", "tokenizers", "semantic_text_splitter", "tesseract")
    paths = [
        Path("src/main.py"),
        Path("src/config.py"),
        Path("src/models.py"),
        Path("src/llama.py"),
        Path("src/rag.py"),
    ]
    documents = Path("src/documents.py")
    if documents.exists():
        paths.append(documents)

    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(name in source for name in banned), path


def test_model_server_microbatches_cover_the_chunk_token_limit() -> None:
    compose = Path("docker-compose.yaml").read_text(encoding="utf-8")
    embed = compose.split("  embed:", 1)[1].split("  rerank:", 1)[0]
    rerank = compose.split("  rerank:", 1)[1]

    assert "--ubatch-size 2048" in embed
    assert "--ubatch-size 2048" in rerank


@pytest.mark.parse_integration
@pytest.mark.skipif(
    os.getenv("RUN_PARSE_INTEGRATION") != "1",
    reason="set RUN_PARSE_INTEGRATION=1 for local LiteParse fixtures",
)
@pytest.mark.parametrize("fixture", [Path("docs/test.pdf"), Path("docs/DACSN.docx")])
def test_local_liteparse_fixture(fixture: Path) -> None:
    chunks = worker.parse_file(
        fixture,
        file_id="integration",
        file_name=fixture.name,
        max_pages=200,
        tokenizer_name="BAAI/bge-m3",
    )

    assert chunks
    assert all(chunk.text and chunk.refs for chunk in chunks)
