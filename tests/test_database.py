import sqlite3
from pathlib import Path

import pytest

from src.database import Database


@pytest.mark.asyncio
async def test_initialize_creates_wal_schema(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)

    await db.initialize()

    names = await db.read(
        lambda conn: {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    )
    assert {"schema_meta", "sessions", "documents", "chunks", "document_jobs"} <= names
    assert await db.read(
        lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0]
    ) == "wal"


@pytest.mark.asyncio
async def test_write_rolls_back_the_entire_callback(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()

    def broken(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s1", "one", 1.0, 1.0),
        )
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        await db.write(broken)

    assert await db.read(
        lambda conn: conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
    ) == 0
