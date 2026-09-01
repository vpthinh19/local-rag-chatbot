"""One-way, non-destructive import of the application's legacy JSON state."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Protocol
from uuid import uuid4

from src.config import Settings
from src.database import Database
from src.models import (
    Chunk,
    DataValidationError,
    Document,
    Message,
    MigrationReport,
)


_CORPUS_MARKER = "legacy_import_v1_corpus"
_HISTORY_MARKER = "legacy_import_v1_history"
_COMPLETE_MARKER = "legacy_import_v1"
_EMBEDDING_SIGNATURE = "embedding_signature"
_LEGACY_SESSION_ID = "legacy-default"
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_MAX_ERROR_CHARS = 500


class _Session(Protocol):
    """The small SDK Session surface required by legacy-history import."""

    async def get_items(self, limit: int | None = None) -> list[Any]: ...

    async def add_items(self, items: list[Any]) -> None: ...


def _error(message: str) -> str:
    """Return a deterministic diagnostic that fits durable error limits."""
    return message.replace("\x00", " ")[:_MAX_ERROR_CHARS]


def _embedding_signature(settings: Settings) -> str:
    """Identify the configured embedding service without probing a model server."""
    return settings.embedding_signature


async def migrate_legacy(
    settings: Settings,
    database: Database,
    session_factory: Callable[[str], _Session],
) -> MigrationReport:
    """Import JSON corpus/history once, then invalidate vectors on signature change."""
    settings.ensure_dirs()
    markers = await database.read(
        lambda conn: {
            str(key): str(value)
            for key, value in conn.execute(
                "SELECT key, value FROM schema_meta WHERE key IN (?, ?, ?)",
                (_CORPUS_MARKER, _HISTORY_MARKER, _COMPLETE_MARKER),
            )
        }
    )
    errors: list[str] = []
    imported_documents = 0
    imported_messages = 0

    if _COMPLETE_MARKER not in markers:
        if _CORPUS_MARKER not in markers:
            count, corpus_errors, corpus_complete = await _import_corpus(settings, database)
            imported_documents += count
            errors.extend(corpus_errors)
            if corpus_complete:
                await _set_marker(database, _CORPUS_MARKER)
        if _HISTORY_MARKER not in markers:
            count, history_errors = await _import_history(settings, database, session_factory)
            imported_messages += count
            errors.extend(history_errors)
            await _set_marker(database, _HISTORY_MARKER)
        await _complete_if_parts_marked(database)

    reindexed_documents = await _invalidate_changed_embeddings(settings, database)
    return MigrationReport(
        imported_documents=imported_documents,
        imported_messages=imported_messages,
        reindexed_documents=reindexed_documents,
        errors=tuple(errors),
    )


async def _set_marker(database: Database, marker: str) -> None:
    await database.write(
        lambda conn: conn.execute(
            "INSERT INTO schema_meta(key, value) VALUES(?, '1') "
            "ON CONFLICT(key) DO NOTHING",
            (marker,),
        )
    )


async def _complete_if_parts_marked(database: Database) -> None:
    def complete(conn: Any) -> None:
        marked = {
            row[0]
            for row in conn.execute(
                "SELECT key FROM schema_meta WHERE key IN (?, ?)",
                (_CORPUS_MARKER, _HISTORY_MARKER),
            )
        }
        if marked == {_CORPUS_MARKER, _HISTORY_MARKER}:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES(?, '1') "
                "ON CONFLICT(key) DO NOTHING",
                (_COMPLETE_MARKER,),
            )

    await database.write(complete)


async def _read_json(path: Path, label: str, errors: list[str]) -> object | None:
    try:
        raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
    except FileNotFoundError:
        return None
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(_error(f"invalid legacy {label} JSON: {exc.msg}"))
        return None


async def _import_corpus(
    settings: Settings, database: Database
) -> tuple[int, list[str], bool]:
    errors: list[str] = []
    raw = await _read_json(settings.legacy_corpus_path, "corpus", errors)
    if raw is None:
        return 0, errors, True
    if not isinstance(raw, dict):
        return 0, [_error("invalid legacy corpus: root must be an object")], True
    raw_documents = raw.get("documents", raw.get("summaries", []))
    raw_chunks = raw.get("chunks", [])
    if not isinstance(raw_documents, list):
        return 0, [_error("invalid legacy corpus: documents must be an array")], True
    if not isinstance(raw_chunks, list):
        return 0, [_error("invalid legacy corpus: chunks must be an array")], True

    chunks_by_document: dict[str, list[Chunk]] = {}
    for index, value in enumerate(raw_chunks):
        try:
            chunk = Chunk.from_dict(value)
        except DataValidationError as exc:
            errors.append(_error(f"legacy chunk {index}: {exc}"))
            continue
        chunks_by_document.setdefault(chunk.file_id, []).append(chunk)

    imported = 0
    complete = True
    known_document_ids: set[str] = set()
    for index, value in enumerate(raw_documents):
        try:
            document = Document.from_dict(value)
            if not _SAFE_ID.fullmatch(document.file_id):
                raise DataValidationError("document.file_id is unsafe")
        except DataValidationError as exc:
            errors.append(_error(f"legacy document {index}: {exc}"))
            continue
        if document.file_id in known_document_ids:
            errors.append(_error(f"legacy document {index}: duplicate document file_id"))
            continue
        known_document_ids.add(document.file_id)
        chunks = sorted(chunks_by_document.get(document.file_id, []), key=lambda item: item.chunk_id)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if (
            len(chunks) != document.chunk_count
            or len(set(chunk_ids)) != len(chunk_ids)
            or any(chunk.file_name != document.file_name for chunk in chunks)
        ):
            errors.append(_error(f"legacy document {index}: chunks do not match metadata"))
            continue
        source_available = await _copy_legacy_source(settings, document, errors)
        if source_available is None:
            complete = False
            continue
        if await _insert_document(database, document, chunks, source_available):
            imported += 1

    for document_id in chunks_by_document:
        if document_id not in known_document_ids:
            errors.append(_error(f"legacy chunks reference unknown document: {document_id}"))
    return imported, errors, complete


async def _copy_legacy_source(
    settings: Settings, document: Document, errors: list[str]
) -> bool | None:
    destination = settings.uploads_dir / document.file_id
    if destination.is_file():
        return True
    if Path(document.file_name).name != document.file_name:
        errors.append(_error(f"legacy source file is missing for {document.file_id}"))
        return False
    source = settings.uploads_dir / f"{document.file_id}_{document.file_name}"
    if not source.is_file():
        errors.append(_error(f"legacy source file is missing for {document.file_id}"))
        return False
    temporary = settings.uploads_dir / f".{document.file_id}.{uuid4().hex}.migration-copy"
    try:
        await asyncio.to_thread(_copy_source_atomically, source, temporary, destination)
    except OSError:
        errors.append(_error(f"legacy source file could not be copied for {document.file_id}"))
        return None
    return True


def _copy_source_atomically(source: Path, temporary: Path, destination: Path) -> None:
    """Copy into this run's private file before atomically claiming the final name."""
    try:
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file():
                raise
    finally:
        temporary.unlink(missing_ok=True)


async def _insert_document(
    database: Database,
    document: Document,
    chunks: list[Chunk],
    source_available: bool,
) -> bool:
    now = time.time()
    media_type = mimetypes.guess_type(document.file_name)[0] or "application/octet-stream"
    error = "" if source_available else "legacy source file is missing"
    status = "processing" if source_available else "failed"

    def insert(conn: Any) -> bool:
        created = conn.execute(
            "INSERT INTO documents(id, file_name, media_type, status, overview, chunk_count, error, created_at, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
            (
                document.file_id,
                document.file_name,
                media_type,
                status,
                document.overview,
                len(chunks),
                error,
                now,
                now,
            ),
        ).rowcount
        if created != 1:
            return False
        conn.executemany(
            "INSERT INTO chunks(document_id, chunk_id, refs_json, text, embedding, embedding_dim) "
            "VALUES(?, ?, ?, ?, NULL, NULL)",
            [
                (
                    document.file_id,
                    chunk.chunk_id,
                    json.dumps(chunk.refs, ensure_ascii=False),
                    chunk.text,
                )
                for chunk in chunks
            ],
        )
        if source_available:
            conn.execute(
                "INSERT INTO document_jobs(id, document_id, operation, state, attempts, next_attempt_at, error, created_at, started_at, finished_at) "
                "VALUES(?, ?, 'reindex', 'queued', 0, ?, '', ?, NULL, NULL)",
                (uuid4().hex, document.file_id, now, now),
            )
        return True

    return await database.write(insert)


async def _import_history(
    settings: Settings,
    database: Database,
    session_factory: Callable[[str], _Session],
) -> tuple[int, list[str]]:
    errors: list[str] = []
    raw = await _read_json(settings.legacy_history_path, "history", errors)
    if raw is None:
        return 0, errors
    if not isinstance(raw, dict) or not isinstance(raw.get("messages", []), list):
        return 0, [_error("invalid legacy history: messages must be an array")]
    messages: list[Message] = []
    for index, value in enumerate(raw["messages"]):
        if not isinstance(value, dict):
            errors.append(_error(f"legacy message {index}: must be an object"))
            continue
        if value.get("role") not in {"user", "assistant"}:
            errors.append(_error(f"legacy message {index}: unsupported role"))
            continue
        try:
            messages.append(Message.from_dict(value))
        except DataValidationError as exc:
            errors.append(_error(f"legacy message {index}: {exc}"))
    if not messages:
        return 0, errors

    session = session_factory(_LEGACY_SESSION_ID)
    existing = await session.get_items()
    imported = 0
    if not existing:
        items = [{"role": message.role, "content": message.content} for message in messages]
        await session.add_items(items)
        imported = len(items)

    title = next(
        (message.content for message in messages if message.role == "user"), messages[0].content
    )[: settings.session_title_chars]
    now = time.time()
    await database.write(
        lambda conn: conn.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(id) DO NOTHING",
            (_LEGACY_SESSION_ID, title, now, now),
        )
    )
    return imported, errors


async def _invalidate_changed_embeddings(settings: Settings, database: Database) -> int:
    signature = _embedding_signature(settings)
    now = time.time()

    def invalidate(conn: Any) -> int:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = ?", (_EMBEDDING_SIGNATURE,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES(?, ?)",
                (_EMBEDDING_SIGNATURE, signature),
            )
            return 0
        if row[0] == signature:
            return 0
        document_ids = [
            str(row[0])
            for row in conn.execute("SELECT id FROM documents WHERE status = 'ready'")
        ]
        for document_id in document_ids:
            conn.execute(
                "UPDATE documents SET status = 'processing', error = '', updated_at = ? WHERE id = ?",
                (now, document_id),
            )
            conn.execute(
                "UPDATE chunks SET embedding = NULL, embedding_dim = NULL WHERE document_id = ?",
                (document_id,),
            )
            conn.execute(
                "INSERT INTO document_jobs(id, document_id, operation, state, attempts, next_attempt_at, error, created_at, started_at, finished_at) "
                "SELECT ?, ?, 'reindex', 'queued', 0, ?, '', ?, NULL, NULL "
                "WHERE NOT EXISTS("
                "SELECT 1 FROM document_jobs WHERE document_id = ? AND operation = 'reindex' "
                "AND state IN ('queued', 'running'))",
                (uuid4().hex, document_id, now, now, document_id),
            )
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = ?",
            (signature, _EMBEDDING_SIGNATURE),
        )
        return len(document_ids)

    return await database.write(invalidate)
