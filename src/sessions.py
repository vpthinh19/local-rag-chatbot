"""Durable Agents SDK sessions with bounded model context."""

from __future__ import annotations

from copy import deepcopy
import inspect
import time
from typing import Any, Mapping, Protocol
from uuid import uuid4

from agents import SQLiteSession
from agents.memory import SessionInputCallback, SessionSettings

from src.config import Settings
from src.database import Database
from src.models import DataValidationError, Message, SessionRecord


_SDK_ITEM_LIMIT = 48
_NEW_SESSION_TITLE = "New chat"


class _Session(Protocol):
    """The public SDK session surface used by the transaction wrapper."""

    session_id: str
    session_settings: SessionSettings | None

    async def get_items(self, limit: int | None = None) -> list[Any]: ...

    async def add_items(self, items: list[Any]) -> None: ...

    async def pop_item(self) -> Any | None: ...

    async def clear_session(self) -> None: ...


async def close_session(session: object) -> None:
    """Close an SDK SQLite session while retaining narrow test-double support."""
    close = getattr(session, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _text(value: object) -> str | None:
    """Extract displayable text from supported Responses message content."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for part in value:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def _visible(item: object) -> tuple[str, str] | None:
    if not isinstance(item, Mapping):
        return None
    role = item.get("role")
    if role not in {"user", "assistant"}:
        return None
    text = _text(item.get("content"))
    return None if text is None else (role, text)


def _complete_turns(items: list[Any]) -> list[list[Any]]:
    """Project user-through-assistant pairs while omitting SDK-internal items."""
    turns: list[list[Any]] = []
    pending: Any | None = None
    for item in items:
        visible = _visible(item)
        if visible is None:
            continue
        role, _ = visible
        if role == "user":
            pending = item
        elif pending is not None:
            turns.append([pending, item])
            pending = None
    return turns


def bounded_session_input(
    history_items: list[Any],
    new_items: list[Any],
    *,
    max_messages: int,
    max_chars: int,
) -> list[Any]:
    """Keep a coherent, bounded history suffix and all current SDK-run items."""
    selected: list[list[Any]] = []
    message_count = 0
    char_count = 0
    for turn in reversed(_complete_turns(history_items)):
        turn_chars = sum(len(visible[1]) for item in turn if (visible := _visible(item)))
        if message_count + len(turn) > max_messages or char_count + turn_chars > max_chars:
            break
        selected.append(turn)
        message_count += len(turn)
        char_count += turn_chars
    selected.reverse()
    return [item for turn in selected for item in turn] + list(new_items)


# Documents the exact public callback shape required by the pinned SDK.
_session_input_callback_type: SessionInputCallback = bounded_session_input  # type: ignore[assignment]


class TransactionalSession:
    """Overlay SDK writes so only a completed addition-only turn becomes durable.

    Public Session methods cannot atomically combine rewinding pre-existing history
    with appending new items. Such a mixed overlay therefore fails closed at commit.
    """

    def __init__(self, delegate: _Session) -> None:
        self._delegate = delegate
        self.session_id = getattr(delegate, "session_id", "")
        self.session_settings = getattr(delegate, "session_settings", None)
        self._pending_items: list[Any] = []
        self._durable_pop_count = 0

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """Return the delegate view with buffered additions and rewinds overlaid."""
        durable = await self._delegate.get_items()
        visible = durable[: len(durable) - self._durable_pop_count]
        items = deepcopy(visible) + deepcopy(self._pending_items)
        if limit is None:
            return items
        return [] if limit == 0 else items[-limit:]

    async def add_items(self, items: list[Any]) -> None:
        """Buffer copies of newly produced SDK items until commit."""
        self._pending_items.extend(deepcopy(items))

    async def pop_item(self) -> Any | None:
        """Overlay a pop without allowing a failed run to mutate durable history."""
        if self._pending_items:
            return deepcopy(self._pending_items.pop())
        durable = await self._delegate.get_items()
        index = len(durable) - self._durable_pop_count - 1
        if index < 0:
            return None
        self._durable_pop_count += 1
        return deepcopy(durable[index])

    async def clear_session(self) -> None:
        """Clear explicit session state; runners use commit/discard for turn writes."""
        self._pending_items.clear()
        self._durable_pop_count = 0
        await self._delegate.clear_session()

    async def commit(self) -> None:
        """Persist one buffered addition batch, never an unatomic durable rewind."""
        if self._durable_pop_count:
            raise RuntimeError(
                "cannot atomically commit durable rewinds with public Session APIs"
            )
        if self._pending_items:
            await self._delegate.add_items(deepcopy(self._pending_items))
            self._pending_items.clear()

    async def discard(self) -> None:
        """Forget the overlay without changing any durable SDK item."""
        self._pending_items.clear()
        self._durable_pop_count = 0


class SessionService:
    """Own application metadata while leaving message rows to the Agents SDK."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self._settings = settings
        self._database = database

    async def create(self) -> SessionRecord:
        """Create independent session metadata with a provisional first-message title."""
        now = time.time()
        record = SessionRecord(uuid4().hex, _NEW_SESSION_TITLE, now, now)
        await self._database.write(
            lambda conn: conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
                (record.id, record.title, record.created_at, record.updated_at),
            )
        )
        return record

    async def list(self) -> list[SessionRecord]:
        """List session metadata newest first without reading SDK-owned tables."""
        return await self._database.read(
            lambda conn: [
                self._record(row)
                for row in conn.execute(
                    "SELECT id, title, created_at, updated_at FROM sessions "
                    "ORDER BY created_at DESC, id DESC"
                )
            ]
        )

    async def get(self, session_id: str) -> SessionRecord | None:
        """Return one metadata record without exposing SDK-owned session rows."""
        row = await self._database.read(
            lambda conn: conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        )
        return None if row is None else self._record(row)

    async def rename(self, session_id: str, title: str) -> SessionRecord:
        """Persist a user-selected title for one existing session."""
        if not isinstance(title, str) or not (title := title.strip()):
            raise DataValidationError("session title must be nonempty")
        now = time.time()

        def rename(conn: Any) -> SessionRecord:
            updated = conn.execute(
                "UPDATE sessions SET title = ?, "
                "updated_at = MAX(?, updated_at + 0.000001) WHERE id = ?",
                (title, now, session_id),
            )
            if updated.rowcount != 1:
                raise DataValidationError("session does not exist")
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            assert row is not None
            return self._record(row)

        return await self._database.write(rename)

    async def messages(self, session_id: str) -> list[Message]:
        """Return only complete visible pairs, leaving internal SDK items private."""
        await self._require(session_id)
        # A negative public SQLiteSession limit returns all items; the session's normal
        # default remains the 48-item bound used by agent runs.
        session = self.sdk_session(session_id)
        try:
            items = await session.get_items(limit=-1)
        finally:
            await close_session(session)
        return [
            Message(role, text)
            for turn in _complete_turns(items)
            for item in turn
            if (visible := _visible(item)) is not None
            for role, text in [visible]
        ]

    async def delete(self, session_id: str) -> None:
        """Clear SDK-owned items before removing only this session's metadata."""
        await self._require(session_id)
        session = self.sdk_session(session_id)
        try:
            await session.clear_session()
        finally:
            await close_session(session)
        await self._database.write(
            lambda conn: conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        )

    async def touch_from_first_message(self, session_id: str, message: str) -> SessionRecord:
        """Set the provisional title from the first user message and update activity."""
        if not isinstance(message, str) or not message:
            raise DataValidationError("session first message must be nonempty")
        title = message[: self._settings.session_title_chars]
        now = time.time()

        def touch(conn: Any) -> SessionRecord:
            updated = conn.execute(
                "UPDATE sessions SET title = CASE WHEN updated_at = created_at THEN ? ELSE title END, "
                "updated_at = MAX(?, updated_at + 0.000001) WHERE id = ?",
                (title, now, session_id),
            )
            if updated.rowcount != 1:
                raise DataValidationError("session does not exist")
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            assert row is not None
            return self._record(row)

        return await self._database.write(touch)

    def sdk_session(self, session_id: str) -> SQLiteSession:
        """Create the pinned SDK session backed by the application database."""
        return SQLiteSession(
            session_id,
            self._settings.database_path,
            session_settings=SessionSettings(limit=_SDK_ITEM_LIMIT),
        )

    async def _require(self, session_id: str) -> None:
        exists = await self._database.read(
            lambda conn: conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        )
        if exists is None:
            raise DataValidationError("session does not exist")

    @staticmethod
    def _record(row: Any) -> SessionRecord:
        return SessionRecord(str(row[0]), str(row[1]), float(row[2]), float(row[3]))
