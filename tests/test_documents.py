import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
from threading import Event

import pytest

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS, Settings
from src.database import Database
from src.documents import DocumentService, LiveCorpus, RequestState
from src.models import Chunk, Corpus, DataValidationError, Document, DocumentRecord
from src.parser import ParserService


FAKE_WORKER = Path(__file__).parent / "helpers" / "fake_parse_worker.py"


class FakeLlama:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.block = False
        self.error: Exception | None = None

    async def complete_chat(
        self, messages: list[dict[str, object]], max_tokens: int, temperature: float
    ) -> str:
        del messages, max_tokens, temperature
        self.started.set()
        if self.error:
            raise self.error
        if self.block:
            await asyncio.Future()
        return "Tổng quan tài liệu"


class FakeRag:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.started = asyncio.Event()
        self.block = False
        self.error: Exception | None = None
        self.install_count = 0

    async def prepare_add(self, chunks: list[Chunk]) -> list[Chunk]:
        self.started.set()
        if self.error:
            raise self.error
        if self.block:
            await asyncio.Future()
        return self.chunks + list(chunks)

    def prepare_remove(self, file_id: str) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.file_id != file_id]

    def prepare_clear(self) -> list[Chunk]:
        return []

    def install(self, candidate: list[Chunk]) -> None:
        self.chunks = list(candidate)
        self.install_count += 1


@dataclass
class Harness:
    settings: Settings
    llama: FakeLlama
    rag: FakeRag
    live: LiveCorpus
    service: DocumentService


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    settings = Settings(
        data_dir=tmp_path / "data",
        max_upload_bytes=1_024,
        parse_termination_grace_seconds=0.05,
    )
    settings.ensure_dirs()
    Corpus().save(settings.corpus_path)
    llama = FakeLlama()
    rag = FakeRag()
    live = LiveCorpus(Corpus())
    service = DocumentService(settings, llama, live, rag)

    async def spawn_fake(command: list[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            str(FAKE_WORKER),
            *command[3:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    monkeypatch.setattr(service, "_spawn_worker", spawn_fake)
    monkeypatch.setenv("FAKE_PARSE_MODE", "success")
    return Harness(settings, llama, rag, live, service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upload_name, expected", [("../../safe name.pdf", "safe name.pdf"), ("doc.docx", "doc.docx")]
)
async def test_successful_ingest_sanitizes_and_commits(
    harness: Harness, upload_name: str, expected: str
) -> None:
    state = RequestState("request-success")

    document = await harness.service.ingest(upload_name, b"content", state)

    assert document.file_name == expected
    assert document.overview == "Tổng quan tài liệu"
    assert harness.live.value.documents == [document]
    assert len(harness.rag.chunks) == 1
    assert Corpus.load(harness.settings.corpus_path) == harness.live.value
    uploads = list(harness.settings.uploads_dir.iterdir())
    assert len(uploads) == 1
    assert uploads[0].name == f"{document.file_id}_{expected}"
    assert uploads[0].read_bytes() == b"content"
    assert not (harness.settings.staging_dir / state.request_id).exists()


@pytest.mark.parametrize("extension", sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
def test_safe_name_accepts_every_liteparse_extension(extension: str) -> None:
    assert DocumentService._safe_name(f"Tài liệu{extension.upper()}") == (
        f"Tài liệu{extension.upper()}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bad.txt", "", "a.pdf\x00evil"])
async def test_ingest_rejects_invalid_name(harness: Harness, name: str) -> None:
    with pytest.raises(DataValidationError):
        await harness.service.ingest(name, b"content", RequestState("bad-name"))


@pytest.mark.asyncio
async def test_ingest_rejects_empty_or_large_content(harness: Harness) -> None:
    with pytest.raises(DataValidationError, match="empty"):
        await harness.service.ingest("a.pdf", b"", RequestState("empty"))
    with pytest.raises(DataValidationError, match="size"):
        await harness.service.ingest("a.pdf", b"x" * 1_025, RequestState("large"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, message", [("fail", "code 7"), ("malformed", "invalid chunks")])
async def test_worker_failure_never_commits(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    message: str,
) -> None:
    monkeypatch.setenv("FAKE_PARSE_MODE", mode)

    with pytest.raises((RuntimeError, DataValidationError), match=message):
        await harness.service.ingest("a.pdf", b"content", RequestState(f"worker-{mode}"))

    assert harness.live.value == Corpus()
    assert harness.rag.install_count == 0
    assert list(harness.settings.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_worker_group_is_killed_and_reaped(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PARSE_MODE", "wait")
    state = RequestState("cancel-worker")
    task = asyncio.create_task(harness.service.ingest("a.pdf", b"content", state))
    pids_path: Path | None = None
    for _ in range(200):
        matches = list(harness.settings.staging_dir.rglob("chunks.pids"))
        if matches and state.parse_process is not None:
            pids_path = matches[0]
            break
        await asyncio.sleep(0.01)
    assert pids_path is not None and pids_path.exists()
    pids = [int(value) for value in pids_path.read_text().split()]
    process = state.parse_process

    state.cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    assert process is not None and process.returncode == -signal_number("KILL")
    assert state.parse_process is None
    assert harness.live.value == Corpus()
    assert not (harness.settings.staging_dir / state.request_id).exists()
    assert not [
        pending
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
        and pending.get_coro().__qualname__ == "Process.wait"
    ]
    for _ in range(200):
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            break
        await asyncio.sleep(0.01)
    assert not any(Path(f"/proc/{pid}").exists() for pid in pids)


@pytest.mark.asyncio
async def test_timed_out_worker_group_is_killed_and_never_commits(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_PARSE_MODE", "wait")
    object.__setattr__(harness.settings, "parse_timeout_seconds", 0.05)
    state = RequestState("timeout-worker")

    with pytest.raises(TimeoutError, match="timed out"):
        await harness.service.ingest("a.pdf", b"content", state)

    assert state.parse_process is None
    assert harness.live.value == Corpus()
    assert harness.rag.install_count == 0
    assert list(harness.settings.uploads_dir.iterdir()) == []
    assert not (harness.settings.staging_dir / state.request_id).exists()


def signal_number(name: str) -> int:
    import signal

    return int(getattr(signal, f"SIG{name}"))


@pytest.mark.asyncio
async def test_cancel_during_overview_rolls_back(harness: Harness) -> None:
    harness.llama.block = True
    state = RequestState("cancel-overview")
    task = asyncio.create_task(harness.service.ingest("a.pdf", b"content", state))
    await asyncio.wait_for(harness.llama.started.wait(), timeout=2)

    state.cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.live.value == Corpus()
    assert harness.rag.install_count == 0
    assert list(harness.settings.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancel_during_candidate_embedding_rolls_back(harness: Harness) -> None:
    harness.rag.block = True
    state = RequestState("cancel-embedding")
    task = asyncio.create_task(harness.service.ingest("a.pdf", b"content", state))
    await asyncio.wait_for(harness.rag.started.wait(), timeout=2)
    assert harness.live.value == Corpus()

    state.cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.live.value == Corpus()
    assert harness.rag.install_count == 0
    assert list(harness.settings.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["overview", "embedding", "persistence"])
async def test_precommit_failures_roll_back(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    if phase == "overview":
        harness.llama.error = RuntimeError("overview failed")
    elif phase == "embedding":
        harness.rag.error = RuntimeError("embedding failed")
    else:
        monkeypatch.setattr(
            Corpus,
            "save",
            lambda self, path: (_ for _ in ()).throw(OSError("save failed")),
        )

    with pytest.raises((RuntimeError, OSError), match="failed"):
        await harness.service.ingest("a.pdf", b"content", RequestState(f"fail-{phase}"))

    assert harness.live.value == Corpus()
    assert harness.rag.install_count == 0
    assert list(harness.settings.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_document_persists_after_ingest_commit(harness: Harness) -> None:
    state = RequestState("commit-then-cancel")
    document = await harness.service.ingest("a.pdf", b"content", state)

    state.cancel_event.set()

    assert harness.live.value.documents == [document]
    assert Corpus.load(harness.settings.corpus_path).documents == [document]
    assert list(harness.settings.uploads_dir.iterdir())


@pytest.mark.asyncio
async def test_legacy_ingest_delegates_parsing_to_parser_service(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, Path]] = []

    async def parse(
        self: ParserService, document_id: str, file_name: str, source_path: Path
    ) -> list[Chunk]:
        del self
        calls.append((document_id, file_name, source_path))
        return [Chunk(document_id, file_name, 0, ["p. 1"], "meaningful text")]

    monkeypatch.setattr(ParserService, "parse", parse)
    document = await harness.service.ingest(
        "report.pdf", b"content", RequestState("parser-service")
    )

    assert len(calls) == 1
    assert calls[0][:2] == (document.file_id, "report.pdf")
    assert calls[0][2].name == "input.pdf"


@pytest.mark.asyncio
async def test_delete_removes_only_selected_document(harness: Harness) -> None:
    first = await harness.service.ingest("first.pdf", b"one", RequestState("first"))
    second = await harness.service.ingest("second.pdf", b"two", RequestState("second"))

    assert harness.service.delete(first.file_id) is True

    assert harness.live.value.documents == [second]
    assert {chunk.file_id for chunk in harness.rag.chunks} == {second.file_id}
    assert not (harness.settings.uploads_dir / f"{first.file_id}_{first.file_name}").exists()
    assert (harness.settings.uploads_dir / f"{second.file_id}_{second.file_name}").exists()


@pytest.mark.asyncio
async def test_delete_save_failure_restores_upload_and_live_state(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = await harness.service.ingest("a.pdf", b"content", RequestState("delete-fail"))
    before = harness.live.value
    upload = harness.settings.uploads_dir / f"{document.file_id}_{document.file_name}"
    monkeypatch.setattr(
        Corpus,
        "save",
        lambda self, path: (_ for _ in ()).throw(OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        harness.service.delete(document.file_id)

    assert upload.exists()
    assert harness.live.value == before
    assert {chunk.file_id for chunk in harness.rag.chunks} == {document.file_id}


def test_startup_prunes_missing_and_orphan_uploads(harness: Harness) -> None:
    present = Document("present", "present.pdf", "", 1)
    missing = Document("missing", "missing.pdf", "", 1)
    corpus = Corpus(
        [present, missing],
        [
            Chunk("present", "present.pdf", 0, [], "present text"),
            Chunk("missing", "missing.pdf", 0, [], "missing text"),
        ],
    )
    (harness.settings.uploads_dir / "present_present.pdf").write_bytes(b"present")
    orphan = harness.settings.uploads_dir / "orphan.pdf"
    orphan.write_bytes(b"orphan")

    pruned = harness.service.prune_missing_uploads(corpus)

    assert pruned.documents == [present]
    assert [chunk.file_id for chunk in pruned.chunks] == ["present"]
    assert not orphan.exists()
    assert Corpus.load(harness.settings.corpus_path) == pruned


@pytest.mark.asyncio
async def test_clear_persists_empty_state_and_removes_uploads(harness: Harness) -> None:
    await harness.service.ingest("a.pdf", b"content", RequestState("clear"))

    harness.service.clear()

    assert harness.live.value == Corpus()
    assert harness.rag.chunks == []
    assert Corpus.load(harness.settings.corpus_path) == Corpus()
    assert list(harness.settings.uploads_dir.iterdir()) == []


class UploadStub:
    """Small async upload double that records the service's stream reads."""

    def __init__(self, filename: str, content: bytes, content_type: str) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content
        self._offset = 0
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._offset >= len(self._content):
            return b""
        stop = len(self._content) if size < 0 else self._offset + size
        value = self._content[self._offset : stop]
        self._offset += len(value)
        return value


async def _durable_documents(tmp_path: Path) -> tuple[Settings, Database, DocumentService]:
    settings = Settings(data_dir=tmp_path / "durable-data")
    settings.ensure_dirs()
    database = Database(settings.database_path, settings.database_busy_timeout_ms)
    await database.initialize()
    return settings, database, DocumentService(settings, database)


async def _job_states(database: Database, document_id: str) -> list[tuple[str, str]]:
    return await database.read(
        lambda conn: [
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT operation, state FROM document_jobs "
                "WHERE document_id = ? ORDER BY created_at, id",
                (document_id,),
            )
        ]
    )


@pytest.mark.asyncio
async def test_upload_commits_processing_job_without_parsing(tmp_path: Path) -> None:
    settings, database, documents = await _durable_documents(tmp_path)
    upload = UploadStub("report.pdf", b"pdf bytes", "application/pdf")

    document = await documents.create_upload(upload)

    assert document.status == "processing"
    assert documents.source_path(document.id) == settings.uploads_dir / document.id
    assert documents.source_path(document.id).read_bytes() == b"pdf bytes"
    assert await _job_states(database, document.id) == [("ingest", "queued")]
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024]


@pytest.mark.asyncio
async def test_durable_document_lookup_and_listing_return_committed_metadata(
    tmp_path: Path,
) -> None:
    _settings, _database, documents = await _durable_documents(tmp_path)
    document = await documents.create_upload(
        UploadStub("report.pdf", b"pdf bytes", "application/pdf")
    )

    assert await documents.get(document.id) == document
    assert await documents.list() == [document]
    assert await documents.get("missing") is None


@pytest.mark.asyncio
async def test_upload_overflow_cleans_its_staging_file(tmp_path: Path) -> None:
    settings, database, documents = await _durable_documents(tmp_path)
    upload = UploadStub(
        "large.pdf", b"x" * (settings.max_upload_bytes + 1), "application/pdf"
    )

    with pytest.raises(DataValidationError, match="size limit"):
        await documents.create_upload(upload)

    assert upload.read_sizes[:2] == [1024 * 1024, 1024 * 1024]
    assert list(settings.staging_dir.iterdir()) == []
    assert list(settings.uploads_dir.iterdir()) == []
    assert await database.read(
        lambda conn: conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    ) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "media_type"),
    [
        ("escape.pdf\x00", "application/pdf"),
        ("report.txt", "text/plain"),
        ("report.pdf", "text/plain"),
    ],
)
async def test_upload_rejects_unsafe_or_unsupported_metadata(
    tmp_path: Path, filename: str, media_type: str
) -> None:
    settings, _database, documents = await _durable_documents(tmp_path)

    with pytest.raises(DataValidationError):
        await documents.create_upload(UploadStub(filename, b"content", media_type))

    assert list(settings.staging_dir.iterdir()) == []
    assert list(settings.uploads_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_database_failure_removes_only_its_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database, documents = await _durable_documents(tmp_path)

    async def fail_write(callback: object) -> object:
        del callback
        raise OSError("database unavailable")

    monkeypatch.setattr(database, "write", fail_write)
    with pytest.raises(OSError, match="database unavailable"):
        await documents.create_upload(
            UploadStub("report.pdf", b"content", "application/pdf")
        )

    assert list(settings.uploads_dir.iterdir()) == []
    assert list(settings.staging_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_cancelled_upload_keeps_a_source_when_its_database_write_commits(
    tmp_path: Path,
) -> None:
    settings, database, documents = await _durable_documents(tmp_path)
    entered_write = Event()
    release_write = Event()
    original_write_sync = database._write_sync

    def commit_after_cancellation(callback: object) -> object:
        def delayed_callback(connection: object) -> object:
            entered_write.set()
            assert release_write.wait(timeout=2.0)
            return callback(connection)  # type: ignore[operator]

        return original_write_sync(delayed_callback)  # type: ignore[arg-type]

    database._write_sync = commit_after_cancellation  # type: ignore[method-assign]
    upload = UploadStub("report.pdf", b"content", "application/pdf")
    task = asyncio.create_task(documents.create_upload(upload))
    assert await asyncio.to_thread(entered_write.wait, 1.0)

    task.cancel()
    release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    records = await documents.list()
    assert len(records) == 1
    assert documents.source_path(records[0].id).read_bytes() == b"content"
    assert await _job_states(database, records[0].id) == [("ingest", "queued")]


@pytest.mark.asyncio
async def test_download_is_denied_after_deletion_is_scheduled(tmp_path: Path) -> None:
    _settings, _database, documents = await _durable_documents(tmp_path)
    document = await documents.create_upload(
        UploadStub("report.pdf", b"content", "application/pdf")
    )

    deleting = await documents.schedule_delete(document.id)

    assert deleting.status == "deleting"
    with pytest.raises(DataValidationError, match="not available"):
        await documents.download_path(document.id)


@pytest.mark.asyncio
async def test_retry_requires_a_failed_document_and_queues_new_ingest(
    tmp_path: Path,
) -> None:
    _settings, database, documents = await _durable_documents(tmp_path)
    document = await documents.create_upload(
        UploadStub("report.pdf", b"content", "application/pdf")
    )

    with pytest.raises(DataValidationError, match="failed"):
        await documents.retry(document.id)

    await database.write(
        lambda conn: conn.execute(
            "UPDATE documents SET status = 'failed', error = 'parse failed' WHERE id = ?",
            (document.id,),
        )
    )
    retried = await documents.retry(document.id)

    assert retried.status == "processing"
    assert retried.error == ""
    assert await _job_states(database, document.id) == [
        ("ingest", "queued"),
        ("ingest", "queued"),
    ]


@pytest.mark.asyncio
async def test_delete_supersedes_queued_ingest_and_reindex_jobs(tmp_path: Path) -> None:
    _settings, database, documents = await _durable_documents(tmp_path)
    document = await documents.create_upload(
        UploadStub("report.pdf", b"content", "application/pdf")
    )
    await database.write(
        lambda conn: conn.execute(
            "INSERT INTO document_jobs("
            "id, document_id, operation, state, attempts, next_attempt_at, error, "
            "created_at, started_at, finished_at"
            ") VALUES(?, ?, 'reindex', 'queued', 0, 1.0, '', 1.0, NULL, NULL)",
            ("reindex-job", document.id),
        )
    )

    deleting = await documents.schedule_delete(document.id)

    assert deleting.status == "deleting"
    assert set(await _job_states(database, document.id)) == {
        ("ingest", "cancelled"),
        ("reindex", "cancelled"),
        ("delete", "queued"),
    }


@pytest.mark.asyncio
async def test_download_rejects_missing_source_and_reconciliation_removes_orphans(
    tmp_path: Path,
) -> None:
    settings, _database, documents = await _durable_documents(tmp_path)
    document = await documents.create_upload(
        UploadStub("report.pdf", b"content", "application/pdf")
    )
    documents.source_path(document.id).unlink()
    orphan = settings.uploads_dir / "orphan"
    orphan.write_bytes(b"orphan")
    stale_staging = settings.staging_dir / "stale.tmp"
    stale_staging.write_bytes(b"stale")

    with pytest.raises(DataValidationError, match="source file is missing"):
        await documents.download_path(document.id)
    await documents.reconcile_files()

    assert not orphan.exists()
    assert list(settings.staging_dir.iterdir()) == []
