"""Contract tests for the isolated parser process boundary."""

import asyncio
import json
from pathlib import Path
import sys

import pytest

from src.config import Settings
from src.models import Chunk, DataValidationError
from src.parser import ParserService, validate_parsed_chunks


FAKE_WORKER = Path(__file__).parent / "helpers" / "fake_parse_worker.py"


def test_semantic_validation_rejects_markup_only_chunks() -> None:
    with pytest.raises(DataValidationError, match="meaningful content"):
        validate_parsed_chunks(
            [Chunk("d", "x.svg", 0, ["p. 1"], "<svg><path/></svg>")]
        )


def test_semantic_validation_accepts_eight_unicode_letters_or_digits() -> None:
    chunks = [Chunk("d", "x.pdf", 0, ["p. 1"], "Nội dung số 123")]

    validate_parsed_chunks(chunks)


@pytest.mark.asyncio
async def test_parser_stages_committed_source_under_its_validated_suffix(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    source = settings.uploads_dir / "document-id"
    source.write_bytes(b"committed source")
    parser = ParserService(settings)

    async def spawn_fake(command: list[str]) -> asyncio.subprocess.Process:
        input_path = Path(command[command.index("--input") + 1])
        assert input_path.name == "input.PDF"
        assert input_path.read_bytes() == b"committed source"
        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(FAKE_WORKER),
            *command[3:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    parser._spawn_worker = spawn_fake  # type: ignore[method-assign]
    chunks = await parser.parse("document-id", "report.PDF", source)

    assert [chunk.text for chunk in chunks] == ["Nội dung từ fake worker."]
    assert list(settings.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_parser_timeout_reaps_the_process_group_and_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        parse_timeout_seconds=0.05,
        parse_termination_grace_seconds=0.05,
    )
    settings.ensure_dirs()
    source = settings.uploads_dir / "document-id"
    source.write_bytes(b"committed source")
    parser = ParserService(settings)
    monkeypatch.setenv("FAKE_PARSE_MODE", "wait")

    async def spawn_fake(command: list[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(FAKE_WORKER),
            *command[3:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    parser._spawn_worker = spawn_fake  # type: ignore[method-assign]
    with pytest.raises(TimeoutError, match="timed out"):
        await parser.parse("document-id", "report.pdf", source)

    assert list(settings.staging_dir.iterdir()) == []
