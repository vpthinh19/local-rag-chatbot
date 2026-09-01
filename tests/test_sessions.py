"""Durable SDK-session metadata and bounded-context contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import pytest_asyncio

from src.config import Settings
from src.database import Database
from src.models import DataValidationError, Message
from src.sessions import SessionService, TransactionalSession, bounded_session_input


def user(text: str) -> dict[str, object]:
    return {"type": "message", "role": "user", "content": text}


def assistant(text: str) -> dict[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def reasoning(text: str) -> dict[str, object]:
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def function_call(name: str) -> dict[str, str]:
    return {"type": "function_call", "call_id": "call", "name": name, "arguments": "{}"}


def function_output(text: str) -> dict[str, str]:
    return {"type": "function_call_output", "call_id": "call", "output": text}


class MemorySession:
    """A small durable-session double whose stored state is observable."""

    def __init__(self, items: list[dict[str, object]] = []) -> None:
        self.items = deepcopy(items)
        self.add_calls: list[list[dict[str, object]]] = []
        self.pop_calls = 0

    async def get_items(self, limit: int | None = None) -> list[dict[str, object]]:
        items = deepcopy(self.items)
        return items if limit is None else items[-limit:]

    async def add_items(self, items: list[dict[str, object]]) -> None:
        copied = deepcopy(items)
        self.add_calls.append(copied)
        self.items.extend(copied)

    async def pop_item(self) -> dict[str, object] | None:
        self.pop_calls += 1
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.items.clear()


class ClosingSession(MemorySession):
    """Session double that records the lifecycle close required by SQLiteSession."""

    def __init__(self, items: list[dict[str, object]] = []) -> None:
        super().__init__(items)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def get_items(self, limit: int | None = None) -> list[dict[str, object]]:
        return await super().get_items(None if limit == -1 else limit)


@pytest_asyncio.fixture
async def session_service(tmp_path: Path) -> SessionService:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    return SessionService(settings, database)


def test_context_keeps_only_complete_recent_visible_turns() -> None:
    """Would fail if stale tool history or an over-budget old turn reached the model."""
    history = [
        user("old " * 100),
        assistant("old answer"),
        reasoning("secret"),
        function_call("search_documents"),
        function_output("large stale chunk"),
        user("recent question"),
        assistant("recent answer"),
    ]

    result = bounded_session_input(
        history, [user("new question")], max_messages=4, max_chars=60
    )

    assert result == [user("recent question"), assistant("recent answer"), user("new question")]


def test_context_never_splits_a_turn_at_item_or_character_boundary() -> None:
    """Would fail if a lone user or assistant item survived a bounded suffix."""
    old = [user("old question"), assistant("old answer")]
    recent = [user("recent question"), assistant("recent answer")]

    assert bounded_session_input(old + recent, [], max_messages=3, max_chars=100) == recent
    assert bounded_session_input(old + recent, [], max_messages=4, max_chars=25) == []


def test_context_keeps_every_current_item_unchanged() -> None:
    """Would fail if current-run function calls or outputs were stripped by the callback."""
    current = [user("new"), function_call("search_documents"), function_output("result")]

    result = bounded_session_input(
        [user("old"), assistant("answer"), reasoning("private")],
        current,
        max_messages=2,
        max_chars=20,
    )

    assert result[-3:] == current
    assert reasoning("private") not in result


@pytest.mark.asyncio
async def test_transactional_session_discards_failed_turn() -> None:
    """Would fail if an incomplete failed turn reached durable SDK storage."""
    durable = MemorySession([user("existing"), assistant("answer")])
    session = TransactionalSession(durable)

    await session.add_items([user("unfinished"), assistant("partial")])
    await session.discard()

    assert await durable.get_items() == [user("existing"), assistant("answer")]


@pytest.mark.asyncio
async def test_transactional_pop_of_existing_history_is_rolled_back_by_discard() -> None:
    """Would fail if a failed SDK rewind removed a pre-existing durable item."""
    original = [user("existing"), assistant("answer")]
    durable = MemorySession(original)
    session = TransactionalSession(durable)

    assert await session.pop_item() == assistant("answer")
    assert await session.get_items() == [user("existing")]
    await session.discard()

    assert await durable.get_items() == original


@pytest.mark.asyncio
async def test_transactional_commit_rejects_mixed_durable_rewind_without_writing() -> None:
    """Would fail if an unatomic mixed commit could delete prior durable history."""
    durable = MemorySession([user("old"), assistant("old answer")])
    session = TransactionalSession(durable)
    await session.pop_item()
    await session.add_items([assistant("replacement")])

    with pytest.raises(RuntimeError, match="cannot atomically commit durable rewinds"):
        await session.commit()

    assert await durable.get_items() == [user("old"), assistant("old answer")]
    assert durable.pop_calls == 0
    assert durable.add_calls == []


@pytest.mark.asyncio
async def test_transactional_commit_add_failure_leaves_prior_history_unchanged() -> None:
    """Would fail if a failed one-call append could alter pre-existing session history."""

    class FailingAddSession(MemorySession):
        async def add_items(self, items: list[dict[str, object]]) -> None:
            del items
            raise OSError("storage unavailable")

    original = [user("old"), assistant("old answer")]
    durable = FailingAddSession(original)
    session = TransactionalSession(durable)
    await session.add_items([user("new question")])

    with pytest.raises(OSError, match="storage unavailable"):
        await session.commit()
    await session.discard()

    assert await durable.get_items() == original


@pytest.mark.asyncio
async def test_transactional_retry_pop_removes_only_pending_input() -> None:
    """Would fail if the SDK retry path rewound durable history instead of its pending input."""
    durable = MemorySession([user("old"), assistant("old answer")])
    session = TransactionalSession(durable)
    retry_input = user("retry question")
    await session.add_items([retry_input])

    assert await session.pop_item() == retry_input
    await session.add_items([retry_input])
    await session.commit()

    assert await durable.get_items() == [user("old"), assistant("old answer"), retry_input]
    assert durable.pop_calls == 0
    assert durable.add_calls == [[retry_input]]


@pytest.mark.asyncio
async def test_session_metadata_title_projection_and_isolation(
    session_service: SessionService,
) -> None:
    """Would fail if session metadata or visible messages could expose another session's data."""
    first = await session_service.create()
    second = await session_service.create()
    long_first_message = "x" * 100

    changed = await session_service.touch_from_first_message(first.id, long_first_message)
    await session_service.sdk_session(first.id).add_items(
        [
            user("first question"),
            reasoning("hidden"),
            function_call("search_documents"),
            function_output("hidden output"),
            assistant("first answer"),
            user("incomplete"),
        ]
    )
    await session_service.sdk_session(second.id).add_items([user("other session"), assistant("other answer")])

    assert changed.title == "x" * 80
    assert await session_service.messages(first.id) == [
        Message("user", "first question"),
        Message("assistant", "first answer"),
    ]
    assert await session_service.messages(second.id) == [
        Message("user", "other session"),
        Message("assistant", "other answer"),
    ]
    assert [record.id for record in await session_service.list()] == [second.id, first.id]
    assert session_service.sdk_session(first.id).session_settings.limit == 48


@pytest.mark.asyncio
async def test_rename_and_delete_clear_only_the_target_session(
    session_service: SessionService,
) -> None:
    """Would fail if deleting metadata left messages behind or cleared a different session."""
    first = await session_service.create()
    second = await session_service.create()
    await session_service.sdk_session(first.id).add_items([user("one"), assistant("one answer")])
    await session_service.sdk_session(second.id).add_items([user("two"), assistant("two answer")])

    renamed = await session_service.rename(first.id, "Saved title")
    await session_service.delete(first.id)

    assert renamed.title == "Saved title"
    assert await session_service.messages(second.id) == [
        Message("user", "two"),
        Message("assistant", "two answer"),
    ]
    assert [record.id for record in await session_service.list()] == [second.id]
    with pytest.raises(DataValidationError, match="session does not exist"):
        await session_service.messages(first.id)


@pytest.mark.asyncio
async def test_messages_and_delete_close_each_sdk_session(
    session_service: SessionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if short-lived API SDK sessions retained SQLite connections."""
    session = await session_service.create()
    message_session = ClosingSession([user("question"), assistant("answer")])
    delete_session = ClosingSession()
    issued = iter([message_session, delete_session])
    monkeypatch.setattr(session_service, "sdk_session", lambda _id: next(issued))

    assert await session_service.messages(session.id) == [
        Message("user", "question"), Message("assistant", "answer")
    ]
    await session_service.delete(session.id)

    assert message_session.close_calls == 1
    assert delete_session.items == []
    assert delete_session.close_calls == 1
