"""Durable document-worker transitions and snapshot-publication contracts."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import httpx
import numpy as np
import pytest
import pytest_asyncio

from src.config import Settings
from src.database import Database
from src.documents import DocumentService
from src.jobs import DocumentWorker
from src.model_clients import ModelHTTPError
from src.models import Chunk, DocumentRecord
from src.rag import IndexSnapshot, RagService, SnapshotStore


class _Parser:
    def __init__(self) -> None:
        self.pause_after_parse = False
        self.paused = asyncio.Event()
        self.resume = asyncio.Event()
        self.block = False
        self.cancel_started = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.block_cancel = False
        self.error: Exception | None = None

    async def parse(self, document_id: str, file_name: str, source_path: Path) -> list[Chunk]:
        del source_path
        if self.block:
            self.paused.set()
            await asyncio.Future()
        if self.error is not None:
            raise self.error
        if self.pause_after_parse:
            self.paused.set()
            await self.resume.wait()
        return [
            Chunk(document_id, file_name, 0, ["p. 1"], "tám ký tự hợp lệ"),
            Chunk(document_id, file_name, 1, ["p. 2"], "nội dung thứ hai"),
        ]

    async def cancel_active(self) -> None:
        if self.block_cancel:
            self.cancel_started.set()
            await self.cancel_release.wait()
        return None


class _Models:
    def __init__(self) -> None:
        self.embed_error: Exception | None = None
        self.embed_calls: list[list[str]] = []

    async def complete_overview(self, file_name: str, chunks: list[Chunk]) -> str:
        del file_name, chunks
        return "overview"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        return [[3.0, 4.0] for _ in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [1.0] * len(documents)


class _InterleavingModels(_Models):
    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Semaphore(1)
        self.first_batch_started = asyncio.Event()
        self.release_first_batch = asyncio.Event()
        self.query_started = asyncio.Event()
        self.release_query = asyncio.Event()
        self._worker_batches = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self.gate:
            self.embed_calls.append(list(texts))
            if texts == ["query"]:
                self.query_started.set()
                await self.release_query.wait()
            else:
                self._worker_batches += 1
                if self._worker_batches == 1:
                    self.first_batch_started.set()
                    await self.release_first_batch.wait()
            return [[3.0, 4.0] for _ in texts]


class _Runtime:
    def __init__(self, tmp_path: Path, *, batch_size: int = 32) -> None:
        self.settings = Settings(
            data_dir=tmp_path / "data",
            embedding_batch_size=batch_size,
            job_retry_base_seconds=0.01,
        )
        self.settings.ensure_dirs()
        self.database = Database(self.settings.database_path, 2_000)
        self.documents = DocumentService(self.settings, self.database)
        self.parser = _Parser()
        self.models = _Models()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.rag = RagService(
            self.models,
            cpu_executor=self.executor,
            embedding_batch_size=batch_size,
            lexical_candidate_limit=24,
            semantic_candidate_limit=24,
            fused_candidate_limit=16,
            final_chunk_limit=5,
        )
        self.snapshots = SnapshotStore(
            IndexSnapshot((), (), np.empty((0, 0), dtype=np.float32), None)
        )
        self.worker = DocumentWorker(
            self.settings,
            self.database,
            self.documents,
            self.parser,
            self.models,
            self.rag,
            self.snapshots,
        )
        self.document_id = "document"

    async def initialize_document(self, *, status: str = "processing") -> None:
        now = time.time()
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.documents.source_path(self.document_id).write_bytes(b"source")

        def insert(conn: object) -> None:
            conn.execute(  # type: ignore[attr-defined]
                "INSERT INTO documents VALUES(?, ?, ?, ?, '', 0, '', ?, ?)",
                (self.document_id, "report.pdf", "application/pdf", status, now, now),
            )
            conn.execute(  # type: ignore[attr-defined]
                "INSERT INTO document_jobs VALUES(?, ?, 'ingest', 'queued', 0, ?, '', ?, NULL, NULL)",
                ("ingest-job", self.document_id, now, now),
            )

        await self.database.write(insert)

    async def insert_job(self, *, state: str, attempts: int) -> None:
        now = time.time()
        await self.database.write(
            lambda conn: conn.execute(
                "INSERT INTO document_jobs VALUES(?, ?, 'delete', ?, ?, ?, '', ?, NULL, NULL)",
                (f"job-{state}", "absent", state, attempts, now, now),
            )
        )

    async def states(self) -> list[str]:
        return await self.database.read(
            lambda conn: [row[0] for row in conn.execute("SELECT state FROM document_jobs ORDER BY id")]
        )

    async def document_status(self) -> str:
        return await self.database.read(
            lambda conn: conn.execute(
                "SELECT status FROM documents WHERE id = ?", (self.document_id,)
            ).fetchone()[0]
        )

    async def close(self) -> None:
        self.executor.shutdown(wait=True)


@pytest_asyncio.fixture
async def tmp_runtime(tmp_path: Path) -> _Runtime:
    runtime = _Runtime(tmp_path)
    await runtime.database.initialize()
    await runtime.initialize_document()
    try:
        yield runtime
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_recover_requeues_running_jobs(tmp_runtime: _Runtime) -> None:
    await tmp_runtime.insert_job(state="running", attempts=1)

    await tmp_runtime.worker.recover()

    assert await tmp_runtime.states() == ["queued", "queued"]


@pytest.mark.asyncio
async def test_failed_ingest_preserves_live_snapshot(tmp_runtime: _Runtime) -> None:
    before = await tmp_runtime.snapshots.capture()
    tmp_runtime.models.embed_error = ValueError("invalid document embedding shape")

    await tmp_runtime.worker.run_one()

    assert await tmp_runtime.snapshots.capture() is before
    assert await tmp_runtime.document_status() == "failed"


@pytest.mark.asyncio
async def test_delete_requested_during_ingest_cannot_republish_document(
    tmp_runtime: _Runtime,
) -> None:
    tmp_runtime.parser.pause_after_parse = True
    running = asyncio.create_task(tmp_runtime.worker.run_one())
    await tmp_runtime.parser.paused.wait()
    await tmp_runtime.documents.schedule_delete(tmp_runtime.document_id)
    tmp_runtime.parser.resume.set()

    await running
    await tmp_runtime.worker.run_one()

    assert tmp_runtime.document_id not in (await tmp_runtime.snapshots.capture()).document_ids
    assert not tmp_runtime.documents.source_path(tmp_runtime.document_id).exists()


@pytest.mark.asyncio
async def test_ingest_persists_normalized_contiguous_float32_embeddings(
    tmp_runtime: _Runtime,
) -> None:
    await tmp_runtime.worker.run_one()

    row = await tmp_runtime.database.read(
        lambda conn: conn.execute(
            "SELECT embedding, embedding_dim FROM chunks WHERE document_id = ? ORDER BY chunk_id",
            (tmp_runtime.document_id,),
        ).fetchone()
    )
    assert row[1] == 2
    vector = np.frombuffer(row[0], dtype=np.float32)
    assert vector.dtype == np.dtype("float32")
    assert vector.flags.c_contiguous
    assert vector.tolist() == pytest.approx([0.6, 0.8])
    assert await tmp_runtime.document_status() == "ready"
    assert tmp_runtime.document_id in (await tmp_runtime.snapshots.capture()).document_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("parser timed out"),
        httpx.TimeoutException("model timed out"),
        httpx.ConnectError("model unavailable"),
        ModelHTTPError("embedding service returned HTTP 429"),
        ModelHTTPError("embedding service returned HTTP 503"),
    ],
)
async def test_transient_failures_retry_at_most_three_times(
    tmp_runtime: _Runtime, error: Exception
) -> None:
    tmp_runtime.parser.error = error
    for _ in range(3):
        await tmp_runtime.worker.run_one()
        await asyncio.sleep(0.02)

    row = await tmp_runtime.database.read(
        lambda conn: conn.execute(
            "SELECT state, attempts FROM document_jobs WHERE id = 'ingest-job'"
        ).fetchone()
    )
    assert tuple(row) == ("failed", 3)


@pytest.mark.asyncio
async def test_deterministic_failures_are_not_retried(tmp_runtime: _Runtime) -> None:
    tmp_runtime.models.embed_error = ValueError("invalid document embedding shape")

    await tmp_runtime.worker.run_one()

    row = await tmp_runtime.database.read(
        lambda conn: conn.execute(
            "SELECT state, attempts FROM document_jobs WHERE id = 'ingest-job'"
        ).fetchone()
    )
    assert tuple(row) == ("failed", 1)


@pytest.mark.asyncio
async def test_document_batches_release_embedding_gate_for_a_waiting_query(
    tmp_path: Path,
) -> None:
    runtime = _Runtime(tmp_path, batch_size=1)
    await runtime.database.initialize()
    await runtime.initialize_document()
    models = _InterleavingModels()
    runtime.models = models
    runtime.rag = RagService(
        models,
        cpu_executor=runtime.executor,
        embedding_batch_size=1,
        lexical_candidate_limit=24,
        semantic_candidate_limit=24,
        fused_candidate_limit=16,
        final_chunk_limit=5,
    )
    runtime.worker = DocumentWorker(
        runtime.settings, runtime.database, runtime.documents, runtime.parser,
        models, runtime.rag, runtime.snapshots,
    )
    try:
        running = asyncio.create_task(runtime.worker.run_one())
        await models.first_batch_started.wait()
        query = asyncio.create_task(models.embed(["query"]))
        await asyncio.sleep(0)
        models.release_first_batch.set()
        await models.query_started.wait()
        assert not running.done()
        models.release_query.set()
        await query
        await running
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cancelled_stop_settles_worker_and_leaves_running_job_for_recovery(
    tmp_runtime: _Runtime,
) -> None:
    tmp_runtime.parser.block = True
    tmp_runtime.parser.block_cancel = True
    tmp_runtime.worker.start()
    await tmp_runtime.parser.paused.wait()

    stopping = asyncio.create_task(tmp_runtime.worker.stop())
    await tmp_runtime.parser.cancel_started.wait()
    stopping.cancel()
    tmp_runtime.parser.cancel_release.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping

    await asyncio.sleep(0)
    assert tmp_runtime.worker._task is None  # noqa: SLF001 - lifecycle invariant
    assert await tmp_runtime.states() == ["running"]


@pytest.mark.asyncio
async def test_failed_delete_is_repaired_by_a_later_delete_request(
    tmp_runtime: _Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await tmp_runtime.documents.schedule_delete(tmp_runtime.document_id)

    async def fail_build(*args: object) -> IndexSnapshot:
        del args
        raise ValueError("delete preparation failed")

    monkeypatch.setattr(tmp_runtime.rag, "build", fail_build)
    await tmp_runtime.worker.run_one()

    assert sorted(await tmp_runtime.states()) == ["cancelled", "failed"]

    monkeypatch.undo()
    await tmp_runtime.documents.schedule_delete(tmp_runtime.document_id)
    await tmp_runtime.documents.schedule_delete(tmp_runtime.document_id)
    delete_jobs = await tmp_runtime.database.read(
        lambda conn: conn.execute(
            "SELECT count(*) FROM document_jobs WHERE document_id = ? "
            "AND operation = 'delete' AND state IN ('queued', 'running')",
            (tmp_runtime.document_id,),
        ).fetchone()[0]
    )
    assert delete_jobs == 1
    await tmp_runtime.worker.run_one()
    assert await tmp_runtime.documents.get(tmp_runtime.document_id) is None


@pytest.mark.asyncio
async def test_candidate_vectors_are_prepared_in_the_cpu_executor(
    tmp_runtime: _Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    original = DocumentWorker._vectors

    def track_vectors(chunks: list[object]) -> np.ndarray:
        threads.append(threading.get_ident())
        return original(chunks)  # type: ignore[arg-type]

    monkeypatch.setattr(DocumentWorker, "_vectors", staticmethod(track_vectors))
    event_loop_thread = threading.get_ident()

    await tmp_runtime.worker.run_one()

    assert threads
    assert set(threads).isdisjoint({event_loop_thread})


@pytest.mark.asyncio
async def test_arbitrary_error_text_is_not_persisted_or_logged(
    tmp_runtime: _Runtime, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "PROMPT-CHUNK-SECRET-DO-NOT-STORE"
    tmp_runtime.parser.error = ValueError(secret)

    await tmp_runtime.worker.run_one()

    error = await tmp_runtime.database.read(
        lambda conn: conn.execute(
            "SELECT error FROM document_jobs WHERE id = 'ingest-job'"
        ).fetchone()[0]
    )
    assert secret not in error
    assert secret not in caplog.text
