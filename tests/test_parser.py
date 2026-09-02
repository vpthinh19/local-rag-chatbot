"""Contract tests for the isolated parser process boundary."""

import asyncio
import json
from pathlib import Path
import sys
import threading

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


def test_parser_output_rejects_chunks_without_source_references(tmp_path: Path) -> None:
    output = tmp_path / "chunks.json"
    output.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "file_id": "d",
                        "file_name": "report.pdf",
                        "chunk_id": 0,
                        "refs": [],
                        "text": "meaningful content",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="source references"):
        ParserService._load_worker_chunks(output, "d", "report.pdf")


def test_parser_output_rejects_blank_source_references(tmp_path: Path) -> None:
    output = tmp_path / "chunks.json"
    output.write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "file_id": "d",
                        "file_name": "report.pdf",
                        "chunk_id": 0,
                        "refs": [" "],
                        "text": "meaningful content",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="chunk.refs"):
        ParserService._load_worker_chunks(output, "d", "report.pdf")


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
async def test_parser_decodes_and_validates_worker_output_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    source = settings.uploads_dir / "document-id"
    source.write_bytes(b"source")
    parser = ParserService(settings)
    threads: list[int] = []
    original = ParserService._load_worker_chunks

    def record(*args: object) -> list[Chunk]:
        threads.append(threading.get_ident())
        return original(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(ParserService, "_load_worker_chunks", staticmethod(record))
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
    monkeypatch.setenv("FAKE_PARSE_MODE", "success")
    event_loop_thread = threading.get_ident()
    await parser.parse("document-id", "report.pdf", source)
    assert threads and threads != [event_loop_thread]


@pytest.mark.asyncio
async def test_cancelled_parser_settles_decoder_before_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    source = settings.uploads_dir / "document-id"
    source.write_bytes(b"source")
    parser = ParserService(settings)
    entered = threading.Event()
    release = threading.Event()
    settled = threading.Event()
    original = ParserService._load_and_validate_worker_chunks

    def blocked(cls: type[ParserService], *args: object) -> list[Chunk]:
        del cls
        entered.set()
        assert release.wait(timeout=2)
        try:
            return original(*args)  # type: ignore[arg-type]
        finally:
            settled.set()

    monkeypatch.setattr(
        ParserService, "_load_and_validate_worker_chunks", classmethod(blocked)
    )

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
    task = asyncio.create_task(parser.parse("document-id", "report.pdf", source))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    cleaned_before_settlement = not any(settings.staging_dir.iterdir())
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(settled.wait, 1)
    assert not cleaned_before_settlement
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
