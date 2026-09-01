"""Short, off-event-loop SQLite transactions for durable application state."""

import asyncio
from collections.abc import Callable
from pathlib import Path
import sqlite3
from typing import TypeVar


Result = TypeVar("Result")


SCHEMA_VERSION = 1

SCHEMA_V1_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS sessions(
      id TEXT PRIMARY KEY, title TEXT NOT NULL,
      created_at REAL NOT NULL, updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents(
      id TEXT PRIMARY KEY, file_name TEXT NOT NULL, media_type TEXT NOT NULL,
      status TEXT NOT NULL CHECK(status IN ('processing','ready','failed','deleting')),
      overview TEXT NOT NULL DEFAULT '', chunk_count INTEGER NOT NULL DEFAULT 0,
      error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks(
      document_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, refs_json TEXT NOT NULL,
      text TEXT NOT NULL, embedding BLOB, embedding_dim INTEGER,
      PRIMARY KEY(document_id, chunk_id),
      FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_jobs(
      id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
      operation TEXT NOT NULL CHECK(operation IN ('ingest','reindex','delete')),
      state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
      attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,
      error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
      started_at REAL, finished_at REAL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS document_jobs_claim
      ON document_jobs(state, next_attempt_at, created_at)
    """,
)


class Database:
    """Configure connections and keep application writes mutually exclusive."""

    def __init__(self, path: Path, busy_timeout_ms: int) -> None:
        self.path = Path(path)
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")
        self.busy_timeout_ms = busy_timeout_ms
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Apply all known schema migrations before accepting application work."""
        async with self._write_lock:
            await asyncio.to_thread(self._initialize_sync)

    async def read(self, fn: Callable[[sqlite3.Connection], Result]) -> Result:
        """Run a configured read callback outside the event loop."""
        return await asyncio.to_thread(self._read_sync, fn)

    async def write(self, fn: Callable[[sqlite3.Connection], Result]) -> Result:
        """Run one immediate transaction, rolling it back on every failure."""
        async with self._write_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._write_sync, fn))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(worker)
                except BaseException:
                    pass
                raise

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        return connection

    def _initialize_sync(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(SCHEMA_V1_STATEMENTS[0])
            row = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            current_version = 0 if row is None else int(row[0])
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(f"database schema {current_version} is newer than supported")
            if current_version < 1:
                for statement in SCHEMA_V1_STATEMENTS[1:]:
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _read_sync(self, fn: Callable[[sqlite3.Connection], Result]) -> Result:
        connection = self._connect()
        try:
            return fn(connection)
        finally:
            connection.close()

    def _write_sync(self, fn: Callable[[sqlite3.Connection], Result]) -> Result:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            result = fn(connection)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
