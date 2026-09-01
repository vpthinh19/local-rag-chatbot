"""Durable, single-task document work and atomic snapshot publication."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import logging
import re
import time
import traceback
from typing import Any, Protocol

import httpx
import numpy as np

from src.config import Settings
from src.database import Database
from src.documents import DocumentService
from src.model_clients import ModelHTTPError
from src.models import Chunk, DataValidationError, DocumentRecord, JobRecord, StoredChunk
from src.rag import RagService, SnapshotStore


_LOG = logging.getLogger(__name__)
_HTTP_STATUS = re.compile(r"\bHTTP\s+(\d{3})\b")


class _Parser(Protocol):
    async def parse(self, document_id: str, file_name: str, source_path: Any) -> list[Chunk]: ...

    async def cancel_active(self) -> None: ...


class _Models(Protocol):
    async def complete_overview(self, file_name: str, chunks: list[Chunk]) -> str: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def sanitize_error(exc: BaseException, limit: int = 500) -> str:
    """Classify a failure without retaining arbitrary parser or model content."""
    del limit
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return "operation timed out"
    if isinstance(exc, httpx.ConnectError):
        return "model connection failed"
    if isinstance(exc, ModelHTTPError):
        match = _HTTP_STATUS.search(str(exc))
        return f"model service returned HTTP {match.group(1)}" if match else "model service failed"
    if isinstance(exc, (DataValidationError, ValueError)):
        return "document validation failed"
    return "document processing failed"


class DocumentWorker:
    """Claim one SQLite job at a time and publish complete RAG snapshots."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        documents: DocumentService,
        parser: _Parser,
        models: _Models,
        rag: RagService,
        snapshots: SnapshotStore,
    ) -> None:
        self._settings = settings
        self._database = database
        self._documents = documents
        self._parser = parser
        self._models = models
        self._rag = rag
        self._snapshots = snapshots
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._documents.set_waker(self.wake)

    async def recover(self) -> None:
        """Make work interrupted by a previous process eligible again."""
        now = time.time()
        await self._database.write(
            lambda conn: conn.execute(
                "UPDATE document_jobs SET state = 'queued', started_at = NULL "
                "WHERE state = 'running'"
            )
        )
        self.wake()

    async def build_ready_snapshot(self) -> Any:
        """Reconstruct the ready persisted corpus before application readiness."""
        return await self._candidate("", include_processing=False)

    def start(self) -> None:
        """Start the sole long-lived worker loop once."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="document-worker")
        self.wake()

    def wake(self) -> None:
        """Prompt the loop to recheck durable work immediately."""
        self._wake_event.set()

    async def stop(self) -> None:
        """Stop claiming work, settle the active operation, and leave it recoverable."""
        self._stopping = True
        self.wake()
        task = self._task
        if task is None:
            return
        task.cancel()
        parser_stop = asyncio.create_task(self._parser.cancel_active())
        cancelled = await self._drain(parser_stop)
        cancelled = await self._drain(task) or cancelled
        self._task = None
        if cancelled:
            raise asyncio.CancelledError

    @staticmethod
    async def _drain(task: asyncio.Task[Any]) -> bool:
        """Await cleanup to settlement even when this caller is cancelled repeatedly."""
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    # A worker/parser task cancelled by this shutdown has settled;
                    # it is not evidence that the caller was cancelled.
                    break
                cancelled = True
                continue
            except BaseException:
                break
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        return cancelled

    async def run_one(self) -> bool:
        """Claim and process one currently eligible job for tests and the loop."""
        if self._stopping:
            return False
        job = await self._claim_one()
        if job is None:
            return False
        try:
            if job.operation == "ingest":
                await self._ingest(job)
            elif job.operation == "reindex":
                await self._reindex(job)
            else:
                await self._delete(job)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._log_failure("document job failed", job.document_id, exc, job.id)
            await self._fail_or_retry(job, exc)
        return True

    async def _run(self) -> None:
        while not self._stopping:
            self._wake_event.clear()
            if await self.run_one():
                continue
            delay = await self._next_delay()
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _claim_one(self) -> JobRecord | None:
        now = time.time()

        def claim(conn: Any) -> JobRecord | None:
            row = conn.execute(
                "SELECT id, document_id, operation, state, attempts, next_attempt_at, "
                "error, created_at, started_at, finished_at FROM document_jobs "
                "WHERE state = 'queued' AND next_attempt_at <= ? "
                "ORDER BY next_attempt_at, created_at LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE document_jobs SET state = 'running', attempts = attempts + 1, "
                "started_at = ?, finished_at = NULL WHERE id = ? AND state = 'queued'",
                (now, row[0]),
            )
            claimed = conn.execute(
                "SELECT id, document_id, operation, state, attempts, next_attempt_at, "
                "error, created_at, started_at, finished_at FROM document_jobs WHERE id = ?",
                (row[0],),
            ).fetchone()
            return None if claimed is None else self._job_record(claimed)

        return await self._database.write(claim)

    async def _next_delay(self) -> float | None:
        row = await self._database.read(
            lambda conn: conn.execute(
                "SELECT next_attempt_at FROM document_jobs WHERE state = 'queued' "
                "ORDER BY next_attempt_at, created_at LIMIT 1"
            ).fetchone()
        )
        if row is None:
            return None
        return max(0.0, float(row[0]) - time.time())

    async def _ingest(self, job: JobRecord) -> None:
        document = await self._document(job.document_id)
        if document is None or document.status == "deleting":
            await self._cancel_if_running(job)
            return
        if document.status != "processing":
            raise DataValidationError("document is not processing")
        chunks = await self._parser.parse(
            document.id, document.file_name, self._documents.source_path(document.id)
        )
        overview = await self._models.complete_overview(document.file_name, chunks)
        vectors = await self._embed_chunks(chunks)
        await self._stage(document, job, chunks, vectors, overview, replace=True)
        candidate = await self._candidate_for_processing(document.id)
        await self._publish_ready(job, document.id, candidate)

    async def _reindex(self, job: JobRecord) -> None:
        document = await self._document(job.document_id)
        if document is None or document.status == "deleting":
            await self._cancel_if_running(job)
            return
        if document.status != "processing":
            raise DataValidationError("document is not processing")
        chunks = await self._chunks_for_document(document.id)
        if not chunks:
            raise DataValidationError("document has no chunks to reindex")
        vectors = await self._embed_chunks(
            [Chunk(chunk.document_id, document.file_name, chunk.chunk_id, list(chunk.refs), chunk.text) for chunk in chunks]
        )
        parsed = [
            Chunk(chunk.document_id, document.file_name, chunk.chunk_id, list(chunk.refs), chunk.text)
            for chunk in chunks
        ]
        await self._stage(document, job, parsed, vectors, document.overview, replace=False)
        candidate = await self._candidate_for_processing(document.id)
        await self._publish_ready(job, document.id, candidate)

    async def _delete(self, job: JobRecord) -> None:
        candidate = await self._candidate_without(job.document_id)
        published = False
        async with self._snapshots.publication_lock:
            now = time.time()

            def complete(conn: Any) -> bool:
                row = conn.execute(
                    "SELECT status FROM documents WHERE id = ?", (job.document_id,)
                ).fetchone()
                state = conn.execute(
                    "SELECT state FROM document_jobs WHERE id = ?", (job.id,)
                ).fetchone()
                if state is None or state[0] != "running":
                    return False
                if row is not None and row[0] != "deleting":
                    return False
                if row is not None:
                    conn.execute("DELETE FROM documents WHERE id = ?", (job.document_id,))
                conn.execute(
                    "UPDATE document_jobs SET state = 'succeeded', error = '', "
                    "finished_at = ? WHERE id = ?", (now, job.id)
                )
                return True

            published = await self._database.write(complete)
            if published:
                self._snapshots.install_locked(candidate)
        if published:
            try:
                self._documents.source_path(job.document_id).unlink(missing_ok=True)
            except OSError as exc:
                self._log_failure("document source cleanup failed", job.document_id, exc)

    async def _embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            raise DataValidationError("document has no chunks")
        batches: list[np.ndarray] = []
        dimension: int | None = None
        loop = asyncio.get_running_loop()
        for start in range(0, len(chunks), self._settings.embedding_batch_size):
            batch = chunks[start : start + self._settings.embedding_batch_size]
            values = await self._models.embed([chunk.text for chunk in batch])
            matrix = await loop.run_in_executor(
                self._rag._cpu_executor,  # noqa: SLF001 - worker shares RAG's bounded CPU gate
                lambda: RagService._normalize_rows(
                    values,
                    expected_rows=len(batch),
                    label="document embedding",
                ),
            )
            if dimension is None:
                dimension = matrix.shape[1]
            elif matrix.shape[1] != dimension:
                raise ValueError("embedding dimension changed between batches")
            batches.append(matrix)
        return await loop.run_in_executor(
            self._rag._cpu_executor,  # noqa: SLF001 - see above
            lambda: np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32),
        )

    async def _stage(
        self,
        document: DocumentRecord,
        job: JobRecord,
        chunks: list[Chunk],
        vectors: np.ndarray,
        overview: str,
        *,
        replace: bool,
    ) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("document embeddings do not align with chunks")
        now = time.time()

        def stage(conn: Any) -> None:
            row = conn.execute(
                "SELECT status FROM documents WHERE id = ?", (document.id,)
            ).fetchone()
            state = conn.execute(
                "SELECT state FROM document_jobs WHERE id = ?", (job.id,)
            ).fetchone()
            if row is None or row[0] == "deleting" or state is None or state[0] != "running":
                return
            if replace:
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document.id,))
            for chunk, vector in zip(chunks, vectors, strict=True):
                payload = np.ascontiguousarray(vector, dtype=np.float32).tobytes()
                conn.execute(
                    "INSERT INTO chunks(document_id, chunk_id, refs_json, text, embedding, embedding_dim) "
                    "VALUES(?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(document_id, chunk_id) DO UPDATE SET refs_json=excluded.refs_json, "
                    "text=excluded.text, embedding=excluded.embedding, embedding_dim=excluded.embedding_dim",
                    (document.id, chunk.chunk_id, json.dumps(list(chunk.refs), ensure_ascii=False), chunk.text, payload, int(vector.size)),
                )
            conn.execute(
                "UPDATE documents SET overview = ?, chunk_count = ?, error = '', updated_at = ? WHERE id = ?",
                (overview, len(chunks), now, document.id),
            )

        await self._database.write(stage)

    async def _publish_ready(self, job: JobRecord, document_id: str, candidate: Any) -> None:
        async with self._snapshots.publication_lock:
            now = time.time()

            def publish(conn: Any) -> bool:
                document = conn.execute(
                    "SELECT status FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                state = conn.execute(
                    "SELECT state FROM document_jobs WHERE id = ?", (job.id,)
                ).fetchone()
                if state is None or state[0] != "running":
                    return False
                if document is None or document[0] == "deleting":
                    conn.execute(
                        "UPDATE document_jobs SET state = 'cancelled', error = 'superseded by delete', "
                        "finished_at = ? WHERE id = ?", (now, job.id)
                    )
                    return False
                if document[0] != "processing":
                    return False
                conn.execute(
                    "UPDATE documents SET status = 'ready', error = '', updated_at = ? WHERE id = ?",
                    (now, document_id),
                )
                conn.execute(
                    "UPDATE document_jobs SET state = 'succeeded', error = '', finished_at = ? WHERE id = ?",
                    (now, job.id),
                )
                return True

            if await self._database.write(publish):
                self._snapshots.install_locked(candidate)

    async def _fail_or_retry(self, job: JobRecord, exc: BaseException) -> None:
        error = sanitize_error(exc)
        transient = self._is_transient(exc)
        now = time.time()

        def finish(conn: Any) -> None:
            document = conn.execute(
                "SELECT status FROM documents WHERE id = ?", (job.document_id,)
            ).fetchone()
            if (
                document is not None
                and document[0] == "deleting"
                and job.operation in {"ingest", "reindex"}
            ):
                conn.execute(
                    "UPDATE document_jobs SET state = 'cancelled', error = 'superseded by delete', "
                    "finished_at = ? WHERE id = ? AND state = 'running'", (now, job.id)
                )
                return
            if transient and job.attempts < self._settings.job_max_attempts:
                delay = self._settings.job_retry_base_seconds * 2 ** (job.attempts - 1)
                conn.execute(
                    "UPDATE document_jobs SET state = 'queued', next_attempt_at = ?, error = ?, "
                    "finished_at = NULL WHERE id = ? AND state = 'running'",
                    (now + delay, error, job.id),
                )
                return
            conn.execute(
                "UPDATE document_jobs SET state = 'failed', error = ?, finished_at = ? "
                "WHERE id = ? AND state = 'running'", (error, now, job.id)
            )
            conn.execute(
                "UPDATE documents SET status = 'failed', error = ?, updated_at = ? "
                "WHERE id = ? AND status = 'processing'", (error, now, job.document_id)
            )

        await self._database.write(finish)
        self.wake()

    async def _cancel_if_running(self, job: JobRecord) -> None:
        now = time.time()
        await self._database.write(
            lambda conn: conn.execute(
                "UPDATE document_jobs SET state = 'cancelled', error = 'superseded by delete', "
                "finished_at = ? WHERE id = ? AND state = 'running'", (now, job.id)
            )
        )

    async def _document(self, document_id: str) -> DocumentRecord | None:
        row = await self._database.read(
            lambda conn: conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, error, "
                "created_at, updated_at FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        )
        return None if row is None else self._document_record(row)

    async def _chunks_for_document(self, document_id: str) -> list[StoredChunk]:
        rows = await self._database.read(
            lambda conn: list(conn.execute(
                "SELECT document_id, chunk_id, refs_json, text, embedding, embedding_dim "
                "FROM chunks WHERE document_id = ? ORDER BY chunk_id", (document_id,)
            ))
        )
        return [self._stored_chunk(row) for row in rows]

    async def _candidate_for_processing(self, document_id: str) -> Any:
        return await self._candidate(document_id, include_processing=True)

    async def _candidate_without(self, document_id: str) -> Any:
        return await self._candidate(document_id, include_processing=False)

    async def _candidate(self, document_id: str, *, include_processing: bool) -> Any:
        statuses = "('ready', 'processing')" if include_processing else "('ready')"
        rows = await self._database.read(
            lambda conn: (
                list(conn.execute(
                    "SELECT id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at "
                    f"FROM documents WHERE status IN {statuses} "
                    + ("AND (status = 'ready' OR id = ?) " if include_processing else "AND id != ? ")
                    + "ORDER BY created_at, id",
                    (document_id,),
                )),
                list(conn.execute(
                    "SELECT c.document_id, c.chunk_id, c.refs_json, c.text, c.embedding, c.embedding_dim "
                    "FROM chunks c JOIN documents d ON d.id = c.document_id "
                    f"WHERE d.status IN {statuses} "
                    + ("AND (d.status = 'ready' OR d.id = ?) " if include_processing else "AND d.id != ? ")
                    + "ORDER BY c.document_id, c.chunk_id",
                    (document_id,),
                )),
            )
        )
        documents = [self._document_record(row) for row in rows[0]]
        if include_processing:
            documents = [
                replace(document, status="ready")
                if document.id == document_id
                else document
                for document in documents
            ]
        chunks = [self._stored_chunk(row) for row in rows[1]]
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            self._rag._cpu_executor,  # noqa: SLF001 - use RAG's bounded CPU executor
            self._vectors,
            chunks,
        )
        return await self._rag.build(documents, chunks, vectors)

    @staticmethod
    def _log_failure(
        message: str, document_id: str, exc: BaseException, job_id: str | None = None
    ) -> None:
        frames = traceback.extract_tb(exc.__traceback__)
        locations = ",".join(
            f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.name}"
            for frame in frames[-4:]
        )
        _LOG.error(
            "%s job_id=%s document_id=%s error_type=%s frames=%s",
            message,
            job_id or "-",
            document_id,
            type(exc).__name__,
            locations or "-",
        )

    @staticmethod
    def _vectors(chunks: list[StoredChunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        rows: list[np.ndarray] = []
        dimension: int | None = None
        for chunk in chunks:
            if chunk.embedding is None or chunk.embedding_dim is None:
                raise DataValidationError("chunk is missing an embedding")
            vector = np.frombuffer(chunk.embedding, dtype=np.float32)
            if vector.size != chunk.embedding_dim:
                raise DataValidationError("chunk embedding dimension is invalid")
            if dimension is None:
                dimension = vector.size
            elif vector.size != dimension:
                raise DataValidationError("chunk embedding dimensions do not match")
            rows.append(vector)
        return np.ascontiguousarray(np.stack(rows), dtype=np.float32)

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.ConnectError)):
            return True
        if not isinstance(exc, ModelHTTPError):
            return False
        cause = exc.__cause__
        if isinstance(cause, (httpx.TimeoutException, httpx.ConnectError)):
            return True
        match = _HTTP_STATUS.search(str(exc))
        return bool(match and (match.group(1) in {"408", "429"} or match.group(1).startswith("5")))

    @staticmethod
    def _document_record(row: Any) -> DocumentRecord:
        return DocumentRecord(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]),
            int(row[5]), str(row[6]), float(row[7]), float(row[8]),
        )

    @staticmethod
    def _stored_chunk(row: Any) -> StoredChunk:
        try:
            refs = tuple(json.loads(str(row[2])))
        except (TypeError, json.JSONDecodeError) as exc:
            raise DataValidationError("chunk references are invalid") from exc
        payload = row[4]
        return StoredChunk(
            str(row[0]), int(row[1]), refs, str(row[3]),
            None if payload is None else bytes(payload),
            None if row[5] is None else int(row[5]),
        )

    @staticmethod
    def _job_record(row: Any) -> JobRecord:
        return JobRecord(
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]),
            float(row[5]), str(row[6]), float(row[7]),
            None if row[8] is None else float(row[8]),
            None if row[9] is None else float(row[9]),
        )
