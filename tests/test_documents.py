from pathlib import Path
from threading import Event
import threading

import asyncio

import pytest

from src.config import Settings
from src.database import Database
from src.documents import DocumentService
from src.models import DataValidationError
from src.models import DocumentRecord
from src.rag import IndexSnapshot, SnapshotStore
import numpy as np


class Upload:
    def __init__(self, name: str, content: bytes, media_type: str = "application/pdf") -> None:
        self.filename = name
        self.content_type = media_type
        self._content = content
        self._offset = 0

    async def read(self, size: int) -> bytes:
        result = self._content[self._offset : self._offset + size]
        self._offset += len(result)
        return result


@pytest.mark.asyncio
async def test_upload_commits_a_processing_document_and_durable_job(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    documents = DocumentService(settings, database)
    document = await documents.create_upload(Upload("report.pdf", b"pdf"))
    assert document.status == "processing"
    assert documents.source_path(document.id).read_bytes() == b"pdf"
    assert await database.read(lambda conn: conn.execute("SELECT operation, state FROM document_jobs WHERE document_id = ?", (document.id,)).fetchone()) == ("ingest", "queued")


@pytest.mark.asyncio
async def test_upload_rejects_untrusted_metadata_before_committing(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    with pytest.raises(DataValidationError):
        await DocumentService(settings, database).create_upload(Upload("bad.txt", b"x", "text/plain"))


@pytest.mark.asyncio
async def test_cancelled_upload_keeps_source_after_its_database_write_commits(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    documents = DocumentService(settings, database)
    entered = Event()
    release = Event()
    original_write = database._write_sync

    def commit_after_cancellation(callback: object) -> object:
        def delayed(connection: object) -> object:
            entered.set()
            assert release.wait(timeout=2)
            return callback(connection)  # type: ignore[operator]

        return original_write(delayed)  # type: ignore[arg-type]

    database._write_sync = commit_after_cancellation  # type: ignore[method-assign]
    task = asyncio.create_task(documents.create_upload(Upload("report.pdf", b"pdf")))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    records = await documents.list()
    assert len(records) == 1
    assert documents.source_path(records[0].id).read_bytes() == b"pdf"
    assert await database.read(
        lambda conn: conn.execute(
            "SELECT operation, state FROM document_jobs WHERE document_id = ?",
            (records[0].id,),
        ).fetchone()
    ) == ("ingest", "queued")


@pytest.mark.asyncio
async def test_document_reads_wait_for_ready_snapshot_publication(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    snapshots = SnapshotStore(IndexSnapshot((), (), np.empty((0, 0), dtype=np.float32), None))
    documents = DocumentService(settings, database, snapshots.publication_lock)
    now = 1.0
    await database.write(lambda conn: conn.execute(
        "INSERT INTO documents VALUES(?, ?, ?, 'processing', '', 0, '', ?, ?)",
        ("document", "report.pdf", "application/pdf", now, now),
    ))

    async with snapshots.publication_lock:
        await database.write(lambda conn: conn.execute("UPDATE documents SET status = 'ready' WHERE id = 'document'"))
        read = asyncio.create_task(documents.list())
        await asyncio.sleep(0)
        assert not read.done()
        snapshots.install_locked(IndexSnapshot(
            (DocumentRecord("document", "report.pdf", "application/pdf", "ready", "", 0, "", now, now),),
            (), np.empty((0, 0), dtype=np.float32), None,
        ))
    assert (await read)[0].status == "ready"


@pytest.mark.asyncio
async def test_upload_writes_blocks_off_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    threads: list[int] = []
    original = DocumentService._write_block

    def record(handle: object, value: bytes) -> None:
        threads.append(threading.get_ident())
        original(handle, value)

    monkeypatch.setattr(DocumentService, "_write_block", staticmethod(record))
    event_loop_thread = threading.get_ident()
    await DocumentService(settings, database).create_upload(Upload("report.pdf", b"pdf"))
    assert threads and event_loop_thread not in threads


@pytest.mark.asyncio
async def test_cancelled_upload_settles_blocking_write_before_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    entered = Event()
    release = Event()
    settled = Event()
    original = DocumentService._write_block

    def blocked(handle: object, value: bytes) -> None:
        entered.set()
        assert release.wait(timeout=2)
        try:
            original(handle, value)
        finally:
            settled.set()

    monkeypatch.setattr(DocumentService, "_write_block", staticmethod(blocked))
    task = asyncio.create_task(
        DocumentService(settings, database).create_upload(Upload("report.pdf", b"pdf"))
    )
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
