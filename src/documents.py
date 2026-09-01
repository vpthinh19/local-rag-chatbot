"""Document staging, disposable parser lifecycle, and atomic corpus commits."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Awaitable, Callable, TypeVar
from uuid import uuid4

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS, Settings
from src.database import Database
from src.llama import LlamaClient
from src.models import Chunk, Corpus, DataValidationError, Document, DocumentRecord
from src.parser import ParserService
from src.rag import RagIndex


_T = TypeVar("_T")
_SAFE_CHAR = re.compile(r"[^\w .()-]+", re.UNICODE)
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MEDIA_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".csv": frozenset({"text/csv", "application/csv"}),
    ".tsv": frozenset({"text/tab-separated-values"}),
    ".doc": frozenset({"application/msword"}),
    ".docm": frozenset(
        {"application/vnd.ms-word.document.macroenabled.12"}
    ),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".key": frozenset({"application/x-iwork-keynote-sffkey"}),
    ".numbers": frozenset({"application/x-iwork-numbers-sffnumbers"}),
    ".odp": frozenset({"application/vnd.oasis.opendocument.presentation"}),
    ".ods": frozenset({"application/vnd.oasis.opendocument.spreadsheet"}),
    ".odt": frozenset({"application/vnd.oasis.opendocument.text"}),
    ".pages": frozenset({"application/x-iwork-pages-sffpages"}),
    ".ppt": frozenset({"application/vnd.ms-powerpoint"}),
    ".pptm": frozenset(
        {"application/vnd.ms-powerpoint.presentation.macroenabled.12"}
    ),
    ".pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
    ".rtf": frozenset({"application/rtf", "text/rtf"}),
    ".xls": frozenset({"application/vnd.ms-excel"}),
    ".xlsm": frozenset(
        {"application/vnd.ms-excel.sheet.macroenabled.12"}
    ),
    ".xlsx": frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    ),
}
for _image_extension in (
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tiff",
    ".webp",
):
    _MEDIA_TYPES[_image_extension] = frozenset({f"image/{_image_extension[1:]}"})
_MEDIA_TYPES[".jpg"] = frozenset({"image/jpeg"})
_MEDIA_TYPES[".svg"] = frozenset({"image/svg+xml"})


@dataclass(slots=True)
class LiveCorpus:
    """Mutable holder for the currently committed corpus snapshot."""

    value: Corpus


@dataclass(slots=True)
class RequestState:
    """Cancellation and process handles owned by one chat request."""

    request_id: str
    # The event reaches cooperative work; task/process handles stop hard work.
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None
    parse_process: asyncio.subprocess.Process | None = None


class DocumentService:
    """Coordinate disposable parsing and transactional corpus updates."""

    def __init__(
        self,
        settings: Settings,
        database_or_llama: Database | LlamaClient,
        live_corpus: LiveCorpus | None = None,
        rag: RagIndex | None = None,
    ) -> None:
        """Bind durable intake, or retain the legacy synchronous ingestion surface."""
        self._settings = settings
        self._database: Database | None = None
        self._parser = ParserService(settings)
        self._llama: LlamaClient | None = None
        self._live: LiveCorpus | None = None
        self._rag: RagIndex | None = None
        self._waker: Callable[[], None] | None = None
        if isinstance(database_or_llama, Database):
            if live_corpus is not None or rag is not None:
                raise TypeError("durable DocumentService accepts only settings and database")
            self._database = database_or_llama
        else:
            if live_corpus is None or rag is None:
                raise TypeError("legacy DocumentService requires corpus and RAG collaborators")
            self._llama = database_or_llama
            self._live = live_corpus
            self._rag = rag

    async def create_upload(self, upload: Any) -> DocumentRecord:
        """Durably accept one bounded source file and queue, but never parse, it."""
        database = self._require_database()
        file_name = self._safe_name(getattr(upload, "filename", None))
        media_type = self._media_type(file_name, getattr(upload, "content_type", None))
        staged_path = await self._stream_to_staging(upload)
        document_id = uuid4().hex
        committed_path = self.source_path(document_id)
        now = time.time()
        document = DocumentRecord(
            document_id,
            file_name,
            media_type,
            "processing",
            "",
            0,
            "",
            now,
            now,
        )
        try:
            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, committed_path)

            def insert(conn: Any) -> DocumentRecord:
                conn.execute(
                    "INSERT INTO documents("
                    "id, file_name, media_type, status, overview, chunk_count, error, "
                    "created_at, updated_at"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.id,
                        document.file_name,
                        document.media_type,
                        document.status,
                        document.overview,
                        document.chunk_count,
                        document.error,
                        document.created_at,
                        document.updated_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO document_jobs("
                    "id, document_id, operation, state, attempts, next_attempt_at, error, "
                    "created_at, started_at, finished_at"
                    ") VALUES(?, ?, 'ingest', 'queued', 0, ?, '', ?, NULL, NULL)",
                    (uuid4().hex, document.id, now, now),
                )
                return document

            result = await database.write(insert)
            self._wake_worker()
            return result
        except asyncio.CancelledError:
            if not await self._document_write_committed(database, document.id):
                committed_path.unlink(missing_ok=True)
            raise
        except BaseException:
            committed_path.unlink(missing_ok=True)
            raise
        finally:
            staged_path.unlink(missing_ok=True)

    async def list(self) -> list[DocumentRecord]:
        """Return all durable documents in stable newest-first order."""
        database = self._require_database()
        return await database.read(
            lambda conn: [
                self._document_record(row)
                for row in conn.execute(
                    "SELECT id, file_name, media_type, status, overview, chunk_count, "
                    "error, created_at, updated_at FROM documents "
                    "ORDER BY created_at DESC, id DESC"
                )
            ]
        )

    async def get(self, document_id: str) -> DocumentRecord | None:
        """Look up durable document metadata without exposing a filesystem path."""
        database = self._require_database()
        row = await database.read(
            lambda conn: conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, "
                "error, created_at, updated_at FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        )
        return None if row is None else self._document_record(row)

    async def retry(self, document_id: str) -> DocumentRecord:
        """Move exactly one failed document back to processing and queue ingestion."""
        database = self._require_database()
        now = time.time()

        def retry_document(conn: Any) -> DocumentRecord:
            row = conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, "
                "error, created_at, updated_at FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise DataValidationError("document does not exist")
            current = self._document_record(row)
            if current.status != "failed":
                raise DataValidationError("only failed documents can be retried")
            conn.execute(
                "UPDATE documents SET status = 'processing', error = '', updated_at = ? "
                "WHERE id = ?",
                (now, document_id),
            )
            conn.execute(
                "INSERT INTO document_jobs("
                "id, document_id, operation, state, attempts, next_attempt_at, error, "
                "created_at, started_at, finished_at"
                ") VALUES(?, ?, 'ingest', 'queued', 0, ?, '', ?, NULL, NULL)",
                (uuid4().hex, document_id, now, now),
            )
            return DocumentRecord(
                current.id,
                current.file_name,
                current.media_type,
                "processing",
                current.overview,
                current.chunk_count,
                "",
                current.created_at,
                now,
            )

        result = await database.write(retry_document)
        self._wake_worker()
        return result

    async def schedule_delete(self, document_id: str) -> DocumentRecord:
        """Atomically hide a document and supersede queued non-delete work."""
        database = self._require_database()
        now = time.time()

        def mark_deleting(conn: Any) -> DocumentRecord:
            def enqueue_delete() -> None:
                conn.execute(
                    "INSERT INTO document_jobs("
                    "id, document_id, operation, state, attempts, next_attempt_at, error, "
                    "created_at, started_at, finished_at"
                    ") VALUES(?, ?, 'delete', 'queued', 0, ?, '', ?, NULL, NULL)",
                    (uuid4().hex, document_id, now, now),
                )

            row = conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, "
                "error, created_at, updated_at FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                raise DataValidationError("document does not exist")
            current = self._document_record(row)
            if current.status == "deleting":
                active = conn.execute(
                    "SELECT 1 FROM document_jobs WHERE document_id = ? "
                    "AND operation = 'delete' AND state IN ('queued', 'running') LIMIT 1",
                    (document_id,),
                ).fetchone()
                if active is None:
                    enqueue_delete()
                return current
            conn.execute(
                "UPDATE document_jobs SET state = 'cancelled', "
                "error = 'superseded by delete', finished_at = ? "
                "WHERE document_id = ? AND operation IN ('ingest', 'reindex') "
                "AND state = 'queued'",
                (now, document_id),
            )
            conn.execute(
                "UPDATE documents SET status = 'deleting', updated_at = ? WHERE id = ?",
                (now, document_id),
            )
            enqueue_delete()
            return DocumentRecord(
                current.id,
                current.file_name,
                current.media_type,
                "deleting",
                current.overview,
                current.chunk_count,
                current.error,
                current.created_at,
                now,
            )

        result = await database.write(mark_deleting)
        self._wake_worker()
        return result

    def set_waker(self, waker: Callable[[], None]) -> None:
        """Register the durable worker's lightweight wake signal."""
        self._waker = waker

    @property
    def parser(self) -> ParserService:
        """Expose the owned parser for application lifecycle settlement."""
        return self._parser

    def _wake_worker(self) -> None:
        if self._waker is not None:
            self._waker()

    async def download_path(self, document_id: str) -> Path:
        """Return a verified committed source path unless deletion has begun."""
        document = await self.get(document_id)
        if document is None:
            raise DataValidationError("document does not exist")
        if document.status == "deleting":
            raise DataValidationError("document is not available for download")
        path = self.source_path(document.id)
        if not path.is_file():
            raise DataValidationError("document source file is missing")
        return path

    async def reconcile_files(self) -> None:
        """Remove abandoned staging data and committed files with no durable record."""
        database = self._require_database()
        self._settings.ensure_dirs()
        for path in self._settings.staging_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        referenced = set(
            await database.read(
                lambda conn: [str(row[0]) for row in conn.execute("SELECT id FROM documents")]
            )
        )
        for path in self._settings.uploads_dir.iterdir():
            if path.is_file() and path.name not in referenced:
                path.unlink(missing_ok=True)

    def source_path(self, document_id: str) -> Path:
        """Derive the only permitted committed source path from an opaque ID."""
        if not isinstance(document_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", document_id
        ):
            raise DataValidationError("document ID is invalid")
        return self._settings.uploads_dir / document_id

    def _require_database(self) -> Database:
        """Reject durable operations when running through the retained legacy facade."""
        if self._database is None:
            raise RuntimeError("durable document storage is not configured")
        return self._database

    @staticmethod
    async def _document_write_committed(database: Database, document_id: str) -> bool:
        """Drain a cancellation-safe commit check for a write that may have committed."""
        check = asyncio.create_task(
            database.read(
                lambda conn: conn.execute(
                    "SELECT 1 FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
                is not None
            )
        )
        try:
            while not check.done():
                try:
                    await asyncio.shield(check)
                except asyncio.CancelledError:
                    # The caller already has a cancellation to re-raise. Drain the
                    # outcome check first so repeated cancellation cannot misclassify
                    # a committed upload as rolled back.
                    continue
            return check.result()
        except BaseException:
            # An unknown outcome must retain the source for startup reconciliation;
            # deleting it could corrupt a transaction that did commit.
            return True

    async def _stream_to_staging(self, upload: Any) -> Path:
        """Copy an upload in fixed-size blocks without retaining its bytes in memory."""
        read = getattr(upload, "read", None)
        if read is None:
            raise TypeError("upload content must be a readable upload")
        self._settings.staging_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._settings.staging_dir,
            prefix="upload-",
            suffix=".tmp",
        )
        path = Path(temporary_name)
        total = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                while True:
                    value = read(_UPLOAD_CHUNK_BYTES)
                    if inspect.isawaitable(value):
                        value = await value
                    if not value:
                        break
                    if not isinstance(value, bytes):
                        raise TypeError("upload reader must return bytes")
                    total += len(value)
                    if total > self._settings.max_upload_bytes:
                        raise DataValidationError("upload exceeds the size limit")
                    handle.write(value)
            if total == 0:
                raise DataValidationError("upload is empty")
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _document_record(row: Any) -> DocumentRecord:
        """Construct the immutable domain record at the SQLite boundary."""
        return DocumentRecord(
            str(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            int(row[5]),
            str(row[6]),
            float(row[7]),
            float(row[8]),
        )

    @staticmethod
    def _media_type(file_name: str, value: object) -> str:
        """Require a normalized, supported MIME type for the validated suffix."""
        if not isinstance(value, str):
            raise DataValidationError("upload media type is invalid")
        media_type = value.split(";", 1)[0].strip().lower()
        allowed = _MEDIA_TYPES.get(Path(file_name).suffix.lower(), frozenset())
        if media_type not in allowed:
            raise DataValidationError("upload media type is unsupported")
        return media_type

    async def ingest(
        self,
        upload_name: str,
        content_or_upload: bytes | Any,
        request_state: RequestState,
    ) -> Document:
        """Parse and commit one upload, or leave no partial document."""
        safe_name = self._safe_name(upload_name)
        extension = Path(safe_name).suffix.lower()
        file_id = uuid4().hex
        staging = self._settings.staging_dir / request_state.request_id
        staged_input = staging / f"input{extension}"
        final_upload = self._upload_path(file_id, safe_name)

        if staging.exists():
            raise DataValidationError("request staging directory already exists")
        staging.mkdir(parents=True)
        try:
            content = await self._read_upload(content_or_upload)
            self._raise_if_cancelled(request_state)
            staged_input.write_bytes(content)

            chunks = await self._parse_legacy(
                file_id, safe_name, staged_input, request_state
            )
            self._raise_if_cancelled(request_state)
            overview = await self._await_or_cancel(
                self._create_overview(safe_name, chunks), request_state
            )
            document = Document(file_id, safe_name, overview, len(chunks))
            candidate_index = await self._await_or_cancel(
                self._rag.prepare_add(chunks), request_state
            )
            candidate_corpus = self._live.value.with_document(document, chunks)
            self._raise_if_cancelled(request_state)

            # Commit order keeps the live index behind durable file and corpus state.
            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staged_input, final_upload)
            try:
                candidate_corpus.save(self._settings.corpus_path)
            except BaseException:
                final_upload.unlink(missing_ok=True)
                raise
            self._rag.install(candidate_index)
            self._live.value = candidate_corpus
            return document
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def delete(self, file_id: str) -> bool:
        """Transactionally remove one document from disk, corpus, and index."""
        document = next(
            (item for item in self._live.value.documents if item.file_id == file_id),
            None,
        )
        if document is None:
            return False
        candidate_corpus = self._live.value.without_document(file_id)
        candidate_index = self._rag.prepare_remove(file_id)
        upload = self._upload_path(document.file_id, document.file_name)
        temporary = self._settings.staging_dir / f"delete-{uuid4().hex}"
        moved = False
        temporary.parent.mkdir(parents=True, exist_ok=True)
        # Moving first allows the upload to be restored if corpus persistence fails.
        if upload.exists():
            os.replace(upload, temporary)
            moved = True
        try:
            candidate_corpus.save(self._settings.corpus_path)
        except BaseException:
            if moved:
                os.replace(temporary, upload)
            raise
        self._rag.install(candidate_index)
        self._live.value = candidate_corpus
        temporary.unlink(missing_ok=True)
        return True

    def clear(self) -> None:
        """Persist an empty corpus, then remove every committed upload."""
        previous = self._live.value
        empty = Corpus()
        candidate_index = self._rag.prepare_clear()
        empty.save(self._settings.corpus_path)
        self._rag.install(candidate_index)
        self._live.value = empty
        for document in previous.documents:
            self._upload_path(document.file_id, document.file_name).unlink(
                missing_ok=True
            )

    def prune_missing_uploads(self, corpus: Corpus) -> Corpus:
        """Reconcile persisted metadata with source files during startup."""
        self._settings.ensure_dirs()
        kept_documents = [
            document
            for document in corpus.documents
            if self._upload_path(document.file_id, document.file_name).is_file()
        ]
        kept_ids = {document.file_id for document in kept_documents}
        pruned = Corpus(
            kept_documents,
            [chunk for chunk in corpus.chunks if chunk.file_id in kept_ids],
        )
        referenced = {
            self._upload_path(document.file_id, document.file_name).resolve()
            for document in kept_documents
        }
        for path in self._settings.uploads_dir.iterdir():
            if path.is_file() and path.resolve() not in referenced:
                path.unlink(missing_ok=True)
        if pruned != corpus:
            pruned.save(self._settings.corpus_path)
        return pruned

    async def _spawn_worker(
        self, command: list[str]
    ) -> asyncio.subprocess.Process:
        """Spawn the parser in a new process group for bounded cleanup."""
        return await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    async def _parse_legacy(
        self,
        document_id: str,
        file_name: str,
        source_path: Path,
        state: RequestState,
    ) -> list[Chunk]:
        """Bridge the old request cancellation state to the shared parser service."""
        # The forwarding hook keeps the existing testable subprocess seam until
        # the legacy request pipeline is removed in Task 10.
        self._parser._spawn_worker = self._spawn_worker
        work = asyncio.create_task(
            self._parser.parse(document_id, file_name, source_path)
        )
        cancellation = asyncio.create_task(state.cancel_event.wait())
        try:
            while True:
                process = self._parser.active_process
                if process is not None:
                    state.parse_process = process
                done, _ = await asyncio.wait(
                    {work, cancellation},
                    timeout=0.01,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if work in done:
                    return await work
                if cancellation in done and state.cancel_event.is_set():
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
                    raise asyncio.CancelledError
        except asyncio.CancelledError:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise
        finally:
            state.parse_process = None
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

    async def _await_or_cancel(
        self, awaitable: Awaitable[_T], state: RequestState
    ) -> _T:
        """Race asynchronous work against the request cancellation event."""
        work = asyncio.ensure_future(awaitable)
        cancellation = asyncio.create_task(state.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {work, cancellation}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancellation in done and state.cancel_event.is_set():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                raise asyncio.CancelledError
            return await work
        except asyncio.CancelledError:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise
        finally:
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

    async def _read_upload(self, source: bytes | Any) -> bytes:
        """Read bytes or an upload stream without exceeding the size limit."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            content = bytes(source)
        else:
            read = getattr(source, "read", None)
            if read is None:
                raise TypeError("upload content must be bytes or a readable upload")
            parts: list[bytes] = []
            size = 0
            while True:
                value = read(1024 * 1024)
                if inspect.isawaitable(value):
                    value = await value
                if not value:
                    break
                if not isinstance(value, bytes):
                    raise TypeError("upload reader must return bytes")
                size += len(value)
                if size > self._settings.max_upload_bytes:
                    raise DataValidationError("upload exceeds the size limit")
                parts.append(value)
            content = b"".join(parts)
        if not content:
            raise DataValidationError("upload is empty")
        if len(content) > self._settings.max_upload_bytes:
            raise DataValidationError("upload exceeds the size limit")
        return content

    async def _create_overview(
        self, file_name: str, chunks: list[Chunk]
    ) -> str:
        """Generate a bounded overview from parsed chunks."""
        sections = [
            f"[{', '.join(chunk.refs)}]\n{chunk.text}" for chunk in chunks
        ]
        context = "\n\n---\n\n".join(sections)[: self._settings.max_context_chars]
        overview = await self._llama.complete_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Tạo overview tiếng Việt ngắn gọn cho tài liệu: tóm tắt, "
                        "dàn ý và các điểm chính, tối đa 300 từ. "
                        "Chỉ dùng nội dung được cung cấp."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu {file_name}:\n\n{context}",
                },
            ],
            max_tokens=768,
            temperature=0.1,
        )
        if not overview.strip():
            raise DataValidationError("overview model returned empty content")
        return overview.strip()

    @staticmethod
    def _safe_name(upload_name: str) -> str:
        """Normalize an upload display name and enforce supported suffixes."""
        if not isinstance(upload_name, str) or "\x00" in upload_name:
            raise DataValidationError("upload filename is invalid")
        basename = Path(upload_name.replace("\\", "/")).name.strip()
        basename = _SAFE_CHAR.sub("_", basename).strip(" .")
        if not basename or basename in {".", ".."}:
            raise DataValidationError("upload filename is empty")
        if len(basename) > 180:
            raise DataValidationError("upload filename is too long")
        if Path(basename).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
            raise DataValidationError(
                f"định dạng file không được hỗ trợ; định dạng hợp lệ: {supported}"
            )
        return basename

    def _upload_path(self, file_id: str, file_name: str) -> Path:
        """Return the committed source path for a document."""
        return self._settings.uploads_dir / f"{file_id}_{file_name}"

    @staticmethod
    def _raise_if_cancelled(state: RequestState) -> None:
        """Stop a transaction before its next irreversible step."""
        if state.cancel_event.is_set():
            raise asyncio.CancelledError
