import asyncio
import sqlite3
from threading import Event
from pathlib import Path

import pytest

import src.database as database_module
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


@pytest.mark.asyncio
async def test_cancelled_write_retains_lock_until_its_thread_settles(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()
    first_started = Event()
    release_first = Event()
    first_worker_settled = Event()
    second_worker_entered = Event()

    def first(conn: sqlite3.Connection) -> None:
        first_started.set()
        assert release_first.wait(timeout=2.0)
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s1", "one", 1.0, 1.0),
        )

    def second(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s2", "two", 2.0, 2.0),
        )

    original_write_sync = db._write_sync

    def observed_write_sync(fn: object) -> object:
        if fn is second:
            second_worker_entered.set()
        try:
            return original_write_sync(fn)  # type: ignore[arg-type]
        finally:
            if fn is first:
                first_worker_settled.set()

    db._write_sync = observed_write_sync  # type: ignore[method-assign]
    first_task = asyncio.create_task(db.write(first))
    assert await asyncio.to_thread(first_started.wait, 1.0)

    first_task.cancel()
    second_task = asyncio.create_task(db.write(second))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.to_thread(second_worker_entered.wait), timeout=0.1
            )
    finally:
        release_first.set()

    with pytest.raises(asyncio.CancelledError):
        await first_task
    assert first_worker_settled.is_set()
    assert await asyncio.to_thread(second_worker_entered.wait, 1.0)
    await second_task


@pytest.mark.asyncio
async def test_repeatedly_cancelled_write_retains_lock_until_its_thread_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()
    first_started = Event()
    release_first = Event()
    first_worker_settled = Event()
    second_worker_entered = Event()
    cleanup_wait_started = Event()

    def first(conn: sqlite3.Connection) -> None:
        first_started.set()
        assert release_first.wait(timeout=2.0)
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s1", "one", 1.0, 1.0),
        )

    def second(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s2", "two", 2.0, 2.0),
        )

    original_write_sync = db._write_sync

    def observed_write_sync(fn: object) -> object:
        if fn is second:
            second_worker_entered.set()
        try:
            return original_write_sync(fn)  # type: ignore[arg-type]
        finally:
            if fn is first:
                first_worker_settled.set()

    original_shield = database_module.asyncio.shield
    shield_calls = 0

    def observed_shield(awaitable: object) -> object:
        nonlocal shield_calls
        shield_calls += 1
        if shield_calls == 2:
            cleanup_wait_started.set()
        return original_shield(awaitable)  # type: ignore[arg-type]

    db._write_sync = observed_write_sync  # type: ignore[method-assign]
    monkeypatch.setattr(database_module.asyncio, "shield", observed_shield)
    first_task = asyncio.create_task(db.write(first))
    assert await asyncio.to_thread(first_started.wait, 1.0)

    first_task.cancel()
    assert await asyncio.to_thread(cleanup_wait_started.wait, 1.0)
    first_task.cancel()
    second_task = asyncio.create_task(db.write(second))
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.to_thread(second_worker_entered.wait), timeout=0.1
            )
    finally:
        release_first.set()

    with pytest.raises(asyncio.CancelledError):
        await first_task
    assert first_worker_settled.is_set()
    assert await asyncio.to_thread(second_worker_entered.wait, 1.0)
    await second_task


@pytest.mark.asyncio
async def test_write_rolls_back_when_callback_raises_base_exception(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()

    class StopWrite(BaseException):
        pass

    def broken(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s1", "one", 1.0, 1.0),
        )
        raise StopWrite()

    with pytest.raises(StopWrite):
        await db.write(broken)

    assert await db.read(
        lambda conn: conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
    ) == 0
