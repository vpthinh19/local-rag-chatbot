"""Durable document storage and document-job scheduling."""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS, Settings
from src.database import Database
from src.models import DataValidationError, DocumentRecord
from src.parser import ParserService


_SAFE_CHAR = re.compile(r"[^\w .()-]+", re.UNICODE)
_DURABLE_SOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MEDIA_TYPES: dict[str, frozenset[str]] = {
    ".pdf": frozenset({"application/pdf"}),
    ".csv": frozenset({"text/csv", "application/csv"}),
    ".tsv": frozenset({"text/tab-separated-values"}),
    ".doc": frozenset({"application/msword"}),
    ".docm": frozenset({"application/vnd.ms-word.document.macroenabled.12"}),
    ".docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    ".key": frozenset({"application/x-iwork-keynote-sffkey"}),
    ".numbers": frozenset({"application/x-iwork-numbers-sffnumbers"}),
    ".odp": frozenset({"application/vnd.oasis.opendocument.presentation"}),
    ".ods": frozenset({"application/vnd.oasis.opendocument.spreadsheet"}),
    ".odt": frozenset({"application/vnd.oasis.opendocument.text"}),
    ".pages": frozenset({"application/x-iwork-pages-sffpages"}),
    ".ppt": frozenset({"application/vnd.ms-powerpoint"}),
    ".pptm": frozenset({"application/vnd.ms-powerpoint.presentation.macroenabled.12"}),
    ".pptx": frozenset({"application/vnd.openxmlformats-officedocument.presentationml.presentation"}),
    ".rtf": frozenset({"application/rtf", "text/rtf"}),
    ".xls": frozenset({"application/vnd.ms-excel"}),
    ".xlsm": frozenset({"application/vnd.ms-excel.sheet.macroenabled.12"}),
    ".xlsx": frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}),
    ".bmp": frozenset({"image/bmp"}),
    ".gif": frozenset({"image/gif"}),
    ".jpeg": frozenset({"image/jpeg"}),
    ".jpg": frozenset({"image/jpeg"}),
    ".png": frozenset({"image/png"}),
    ".tiff": frozenset({"image/tiff"}),
    ".webp": frozenset({"image/webp"}),
    ".svg": frozenset({"image/svg+xml"}),
}


class DocumentService:
    """Accept files independently from chat and queue only durable work."""

    def __init__(self, settings: Settings, database: Database, publication_lock: Any | None = None) -> None:
        self._settings = settings
        self._database = database
        self._parser = ParserService(settings)
        self._publication_lock = publication_lock or asyncio.Lock()
        self._waker: Callable[[], None] | None = None

    async def create_upload(self, upload: Any) -> DocumentRecord:
        file_name = self._safe_name(getattr(upload, "filename", None))
        media_type = self._media_type(file_name, getattr(upload, "content_type", None))
        staged_path = await self._stream_to_staging(upload)
        document_id = uuid4().hex
        committed_path = self.source_path(document_id)
        now = time.time()
        document = DocumentRecord(
            document_id, file_name, media_type, "processing", "", 0, "", now, now
        )
        try:
            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, committed_path)

            def insert(conn: Any) -> DocumentRecord:
                conn.execute(
                    "INSERT INTO documents(id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.id, document.file_name, document.media_type,
                        document.status, document.overview, document.chunk_count,
                        document.error, document.created_at, document.updated_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO document_jobs(id, document_id, operation, state, attempts, next_attempt_at, error, created_at, started_at, finished_at) "
                    "VALUES(?, ?, 'ingest', 'queued', 0, ?, '', ?, NULL, NULL)",
                    (uuid4().hex, document.id, now, now),
                )
                return document

            result = await self._database.write(insert)
            self._wake_worker()
            return result
        except asyncio.CancelledError:
            if not await self._committed(document.id):
                committed_path.unlink(missing_ok=True)
            raise
        except BaseException:
            committed_path.unlink(missing_ok=True)
            raise
        finally:
            staged_path.unlink(missing_ok=True)

    async def list(self) -> list[DocumentRecord]:
        async with self._publication_lock:
            return await self._database.read(
                lambda conn: [
                    self._record(row)
                    for row in conn.execute(
                        "SELECT id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at "
                        "FROM documents ORDER BY created_at DESC, id DESC"
                    )
                ]
            )

    async def get(self, document_id: str) -> DocumentRecord | None:
        async with self._publication_lock:
            row = await self._database.read(
                lambda conn: conn.execute(
                    "SELECT id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at "
                    "FROM documents WHERE id = ?", (document_id,)
                ).fetchone()
            )
        return None if row is None else self._record(row)

    async def retry(self, document_id: str) -> DocumentRecord:
        now = time.time()

        def retry_document(conn: Any) -> DocumentRecord:
            row = conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at "
                "FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise DataValidationError("document does not exist")
            current = self._record(row)
            if current.status != "failed":
                raise DataValidationError("only failed documents can be retried")
            conn.execute("UPDATE documents SET status = 'processing', error = '', updated_at = ? WHERE id = ?", (now, document_id))
            conn.execute(
                "INSERT INTO document_jobs(id, document_id, operation, state, attempts, next_attempt_at, error, created_at, started_at, finished_at) "
                "VALUES(?, ?, 'ingest', 'queued', 0, ?, '', ?, NULL, NULL)",
                (uuid4().hex, document_id, now, now),
            )
            return DocumentRecord(current.id, current.file_name, current.media_type, "processing", current.overview, current.chunk_count, "", current.created_at, now)

        result = await self._database.write(retry_document)
        self._wake_worker()
        return result

    async def schedule_delete(self, document_id: str) -> DocumentRecord:
        now = time.time()

        def mark_deleting(conn: Any) -> DocumentRecord:
            row = conn.execute(
                "SELECT id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at "
                "FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                raise DataValidationError("document does not exist")
            current = self._record(row)
            active = conn.execute(
                "SELECT 1 FROM document_jobs WHERE document_id = ? AND operation = 'delete' "
                "AND state IN ('queued', 'running')", (document_id,)
            ).fetchone()
            if current.status == "deleting" and active is not None:
                return current
            conn.execute(
                "UPDATE document_jobs SET state = 'cancelled', error = 'superseded by delete', finished_at = ? "
                "WHERE document_id = ? AND operation IN ('ingest', 'reindex') AND state = 'queued'",
                (now, document_id),
            )
            conn.execute("UPDATE documents SET status = 'deleting', updated_at = ? WHERE id = ?", (now, document_id))
            if active is None:
                conn.execute(
                    "INSERT INTO document_jobs(id, document_id, operation, state, attempts, next_attempt_at, error, created_at, started_at, finished_at) "
                    "VALUES(?, ?, 'delete', 'queued', 0, ?, '', ?, NULL, NULL)",
                    (uuid4().hex, document_id, now, now),
                )
            return DocumentRecord(current.id, current.file_name, current.media_type, "deleting", current.overview, current.chunk_count, current.error, current.created_at, now)

        result = await self._database.write(mark_deleting)
        self._wake_worker()
        return result

    def set_waker(self, waker: Callable[[], None]) -> None:
        self._waker = waker

    @property
    def parser(self) -> ParserService:
        return self._parser

    async def download_path(self, document_id: str) -> Path:
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
        self._settings.ensure_dirs()
        for path in self._settings.staging_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        referenced = set(await self._database.read(lambda conn: [str(row[0]) for row in conn.execute("SELECT id FROM documents")]))
        for path in self._settings.uploads_dir.iterdir():
            if path.is_file() and _DURABLE_SOURCE_ID.fullmatch(path.name) and path.name not in referenced:
                path.unlink(missing_ok=True)

    def source_path(self, document_id: str) -> Path:
        if not isinstance(document_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", document_id):
            raise DataValidationError("document ID is invalid")
        return self._settings.uploads_dir / document_id

    async def _committed(self, document_id: str) -> bool:
        check = asyncio.create_task(self._database.read(lambda conn: conn.execute("SELECT 1 FROM documents WHERE id = ?", (document_id,)).fetchone() is not None))
        try:
            while not check.done():
                try:
                    await asyncio.shield(check)
                except asyncio.CancelledError:
                    continue
            return check.result()
        except BaseException:
            return True

    async def _stream_to_staging(self, upload: Any) -> Path:
        read = getattr(upload, "read", None)
        if read is None:
            raise TypeError("upload content must be a readable upload")
        self._settings.staging_dir.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self._settings.staging_dir, prefix="upload-", suffix=".tmp")
        path = Path(name)
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
                    await asyncio.to_thread(self._write_block, handle, value)
            if not total:
                raise DataValidationError("upload is empty")
            return path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    def _wake_worker(self) -> None:
        if self._waker is not None:
            self._waker()

    @staticmethod
    def _write_block(handle: Any, value: bytes) -> None:
        handle.write(value)

    @staticmethod
    def _record(row: Any) -> DocumentRecord:
        return DocumentRecord(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4]), int(row[5]), str(row[6]), float(row[7]), float(row[8]))

    @staticmethod
    def _safe_name(upload_name: object) -> str:
        if not isinstance(upload_name, str) or "\x00" in upload_name:
            raise DataValidationError("upload filename is invalid")
        basename = Path(upload_name.replace("\\", "/")).name.strip()
        basename = _SAFE_CHAR.sub("_", basename).strip(" .")
        if not basename or basename in {".", ".."}:
            raise DataValidationError("upload filename is empty")
        if len(basename) > 180:
            raise DataValidationError("upload filename is too long")
        if Path(basename).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            raise DataValidationError("upload filename has an unsupported extension")
        return basename

    @staticmethod
    def _media_type(file_name: str, value: object) -> str:
        if not isinstance(value, str):
            raise DataValidationError("upload media type is invalid")
        media_type = value.split(";", 1)[0].strip().lower()
        if media_type not in _MEDIA_TYPES.get(Path(file_name).suffix.lower(), frozenset()):
            raise DataValidationError("upload media type is unsupported")
        return media_type
