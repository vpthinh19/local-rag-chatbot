import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

from src.config import Settings
from src.documents import DocumentService, LiveCorpus, RequestState
from src.models import Chunk, Corpus, DataValidationError, Document


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
    pids_path = harness.settings.staging_dir / state.request_id / "chunks.pids"
    for _ in range(200):
        if pids_path.exists() and state.parse_process is not None:
            break
        await asyncio.sleep(0.01)
    assert pids_path.exists()
    pids = [int(value) for value in pids_path.read_text().split()]
    process = state.parse_process

    state.cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)

    assert process is not None and process.returncode == -signal_number("KILL")
    assert state.parse_process is None
    assert harness.live.value == Corpus()
    assert not (harness.settings.staging_dir / state.request_id).exists()
    for _ in range(200):
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            break
        await asyncio.sleep(0.01)
    assert not any(Path(f"/proc/{pid}").exists() for pid in pids)


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
