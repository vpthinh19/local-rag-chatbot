"""Agents SDK tool and streamed-turn contracts."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from agents.tool_context import ToolContext

from src.config import Settings
from src.models import DocumentRecord, StoredChunk
from src.rag import IndexSnapshot, RagService, SnapshotStore
from src.sessions import TransactionalSession


class _Models:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [float(len(documents) - index) for index in range(len(documents))]


class _Session:
    def __init__(self) -> None:
        self.items: list[Any] = []
        self.close_calls = 0

    async def get_items(self, limit: int | None = None) -> list[Any]:
        return self.items if limit is None else self.items[-limit:]

    async def add_items(self, items: list[Any]) -> None:
        self.items.extend(items)

    async def pop_item(self) -> Any | None:
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.items.clear()

    def close(self) -> None:
        self.close_calls += 1


class _Sessions:
    def __init__(self) -> None:
        self.sessions: dict[str, _Session] = {}

    def sdk_session(self, session_id: str) -> _Session:
        return self.sessions.setdefault(session_id, _Session())


class _CompletedStream:
    """Minimal Runner result double that persists one complete SDK turn."""

    def __init__(self, session: TransactionalSession, answer: str = "Câu trả lời") -> None:
        self._session = session
        self.final_output = answer
        self.is_complete = False
        self.cancelled = False
        self.max_turns: int | None = None
        self.run_config: Any = None

    def cancel(self) -> None:
        self.cancelled = True
        # Match Agents SDK RunResultStreaming.cancel(mode="immediate") exactly.
        self.is_complete = True

    async def stream_events(self):
        await self._session.add_items(
            [
                {"type": "message", "role": "user", "content": "question"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": self.final_output}],
                },
            ]
        )
        if self.cancelled:
            return
        self.is_complete = True
        yield type(
            "Raw", (), {"type": "raw_response_event", "data": type("Delta", (), {"type": "response.output_text.delta", "delta": self.final_output})()}
        )()


class _BackgroundStream(_CompletedStream):
    """A result that settles only when its cancelled iterator is drained."""

    def __init__(self, session: TransactionalSession) -> None:
        super().__init__(session)
        self.cancel_calls = 0
        self.background_started = asyncio.Event()
        self.cancel_requested = asyncio.Event()
        self.allow_settlement = asyncio.Event()
        self.iterator_settled = asyncio.Event()
        self._background_task = asyncio.create_task(self._run_background())

    async def _run_background(self) -> None:
        await asyncio.Event().wait()

    def cancel(self) -> None:
        self.cancel_calls += 1
        super().cancel()
        self.cancel_requested.set()
        self._background_task.cancel()

    def stream_events(self) -> "_DrainOnlyEvents":
        return _DrainOnlyEvents(self)


class _DrainOnlyEvents:
    """Models an SDK iterator whose cancellation finalizer runs only on a later next()."""

    def __init__(self, result: _BackgroundStream) -> None:
        self._result = result
        self._first = True

    def __aiter__(self) -> "_DrainOnlyEvents":
        return self

    async def __anext__(self) -> object:
        if self._first:
            self._first = False
            await self._result._session.add_items(
                [
                    {"type": "message", "role": "user", "content": "question"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self._result.final_output}],
                    },
                ]
            )
            self._result.background_started.set()
            return type(
                "Raw",
                (),
                {
                    "type": "raw_response_event",
                    "data": type(
                        "Delta",
                        (),
                        {"type": "response.output_text.delta", "delta": self._result.final_output},
                    )(),
                },
            )()
        if self._result.cancelled:
            await self._result.allow_settlement.wait()
            await asyncio.gather(self._result._background_task, return_exceptions=True)
            self._result.iterator_settled.set()
            raise StopAsyncIteration
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _OwnerCancelledStream(_CompletedStream):
    """Models an SDK iterator that cannot be drained after its owning task is cancelled."""

    def __init__(self, session: TransactionalSession) -> None:
        super().__init__(session)
        self.cancel_requested = asyncio.Event()
        self.allow_settlement = asyncio.Event()
        self.run_loop_settled = asyncio.Event()
        self._background_task = asyncio.create_task(self._run_background())
        self.events = _OwnerCancelledEvents(self)

    async def _run_background(self) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await self.allow_settlement.wait()
            self.run_loop_settled.set()
            raise

    def cancel(self) -> None:
        super().cancel()
        self.cancel_requested.set()
        self._background_task.cancel()

    def stream_events(self) -> "_OwnerCancelledEvents":
        return self.events


class _OwnerCancelledEvents:
    def __init__(self, result: _OwnerCancelledStream) -> None:
        self._result = result
        self._first = True
        self.next_started = asyncio.Event()
        self.owner_cancelled = False
        self.iterator_settled = asyncio.Event()

    def __aiter__(self) -> "_OwnerCancelledEvents":
        return self

    async def __anext__(self) -> object:
        if self.owner_cancelled:
            raise StopAsyncIteration
        if self._first:
            self._first = False
            await self._result._session.add_items(
                [
                    {"type": "message", "role": "user", "content": "question"},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self._result.final_output}],
                    },
                ]
            )
            return type(
                "Raw",
                (),
                {
                    "type": "raw_response_event",
                    "data": type(
                        "Delta",
                        (),
                        {"type": "response.output_text.delta", "delta": self._result.final_output},
                    )(),
                },
            )()
        self.next_started.set()
        try:
            await self._result._background_task
        except asyncio.CancelledError:
            if not self._result.cancelled:
                self.owner_cancelled = True
                raise
            self.iterator_settled.set()
            raise StopAsyncIteration


class _BarrierCompletedStream(_CompletedStream):
    """A normally completed result whose final SDK event is test-controlled."""

    def __init__(self, session: TransactionalSession) -> None:
        super().__init__(session)
        self.cancel_calls = 0
        self.waiting_to_finish = asyncio.Event()
        self.allow_finish = asyncio.Event()

    def cancel(self) -> None:
        self.cancel_calls += 1
        super().cancel()

    async def stream_events(self):
        self.waiting_to_finish.set()
        await self.allow_finish.wait()
        self.is_complete = True
        if False:
            yield None


class _QueuedLateStream(_CompletedStream):
    """Models an SDK queue with a delta already enqueued when stop is requested."""

    async def stream_events(self):
        await self._session.add_items(
            [{"type": "message", "role": "user", "content": "question"}]
        )
        yield type(
            "Raw",
            (),
            {
                "type": "raw_response_event",
                "data": type("Delta", (), {"type": "response.output_text.delta", "delta": "early"})(),
            },
        )()
        yield type(
            "Raw",
            (),
            {
                "type": "raw_response_event",
                "data": type("Delta", (), {"type": "response.output_text.delta", "delta": "late"})(),
            },
        )()


class _CommitBarrierSession(_Session):
    """The durable add completes only after the test releases the commit window."""

    def __init__(self) -> None:
        super().__init__()
        self.commit_started = asyncio.Event()
        self.allow_commit = asyncio.Event()

    async def add_items(self, items: list[Any]) -> None:
        self.commit_started.set()
        await self.allow_commit.wait()
        self.items.extend(items)


def _document(document_id: str, *, status: str = "ready", overview: str = "Tóm tắt") -> DocumentRecord:
    return DocumentRecord(document_id, f"{document_id}.pdf", "application/pdf", status, overview, 1, "", 1.0, 1.0)


def _snapshot(document_id: str, *, text: str = "Nội dung chuẩn", overview: str = "Tóm tắt") -> IndexSnapshot:
    document = _document(document_id, overview=overview)
    chunk = StoredChunk(document_id, 0, ("p. 1",), text, b"", 2)
    return IndexSnapshot((document,), (chunk,), np.array([[1.0, 0.0]], dtype=np.float32), None)


def _tool_context(context: Any, tool_name: str, arguments: str) -> ToolContext[Any]:
    return ToolContext(context, tool_name=tool_name, tool_call_id="test", tool_arguments=arguments)


@pytest.fixture
def agent_harness(tmp_path: Path):
    from src.agent import AgentService

    executor = ThreadPoolExecutor(max_workers=1)
    store = SnapshotStore(_snapshot("old"))
    rag = RagService(
        _Models(),
        cpu_executor=executor,
        embedding_batch_size=8,
        lexical_candidate_limit=4,
        semantic_candidate_limit=4,
        fused_candidate_limit=4,
        final_chunk_limit=4,
    )
    sessions = _Sessions()
    service = AgentService(
        Settings(data_dir=tmp_path / "data"), store, rag, sessions, responses_model="local"
    )
    yield type("Harness", (), {"store": store, "rag": rag, "sessions": sessions, "service": service})()
    executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_tools_read_the_run_snapshot_even_after_publication(agent_harness) -> None:
    """Would fail if a tool recaptured a newer live snapshot mid-run."""
    from src.agent import AgentContext, get_document_overviews

    old = await agent_harness.store.capture()
    async with agent_harness.store.publication_lock:
        agent_harness.store.install_locked(_snapshot("new"))

    value = json.loads(
        await get_document_overviews.on_invoke_tool(
            _tool_context(AgentContext(old, agent_harness.rag), "get_document_overviews", json.dumps({"file_ids": ["old"]})),
            json.dumps({"file_ids": ["old"]}),
        )
    )

    assert value == {
        "documents": [{"file_id": "old", "file_name": "old.pdf", "overview": "Tóm tắt"}]
    }


@pytest.mark.asyncio
async def test_tools_reject_unknown_and_unready_document_ids(agent_harness) -> None:
    """Would fail if a tool exposed data outside the captured ready snapshot."""
    from src.agent import AgentContext, get_document_overviews

    context = _tool_context(AgentContext(await agent_harness.store.capture(), agent_harness.rag), "get_document_overviews", json.dumps({"file_ids": ["missing"]}))
    with pytest.raises(ValueError, match="not available"):
        await get_document_overviews.on_invoke_tool(context, json.dumps({"file_ids": ["missing"]}))

    unready = IndexSnapshot((_document("waiting", status="processing"),), (StoredChunk("waiting", 0, ("p. 1",), "Text", b"", 2),), np.array([[1.0, 0.0]], dtype=np.float32), None)
    with pytest.raises(ValueError, match="not available"):
        await get_document_overviews.on_invoke_tool(
            _tool_context(AgentContext(unready, agent_harness.rag), "get_document_overviews", json.dumps({"file_ids": ["waiting"]})),
            json.dumps({"file_ids": ["waiting"]}),
        )


def test_tool_schemas_bound_queries_file_ids_and_limit() -> None:
    """Would fail if the model could request unbounded retrieval inputs."""
    from src.agent import get_document_overviews, search_documents

    search = search_documents.params_json_schema
    overview = get_document_overviews.params_json_schema
    assert search["properties"]["queries"]["minItems"] == 1
    assert search["properties"]["queries"]["maxItems"] == 3
    assert search["properties"]["file_ids"]["minItems"] == 1
    assert search["properties"]["file_ids"]["maxItems"] == 8
    assert search["properties"]["limit"]["minimum"] == 1
    assert search["properties"]["limit"]["maximum"] == 6
    assert overview["properties"]["file_ids"]["minItems"] == 1
    assert overview["properties"]["file_ids"]["maxItems"] == 8


@pytest.mark.asyncio
async def test_tool_result_fails_closed_before_cutting_json(agent_harness) -> None:
    """Would fail if an oversized tool result were truncated into invalid JSON."""
    from src.agent import AgentContext, get_document_overviews

    oversized = _snapshot("old", overview="x" * 48_000)
    with pytest.raises(ValueError, match="too large"):
        await get_document_overviews.on_invoke_tool(
                _tool_context(AgentContext(oversized, agent_harness.rag), "get_document_overviews", json.dumps({"file_ids": ["old"]})),
            json.dumps({"file_ids": ["old"]}),
        )


@pytest.mark.asyncio
async def test_search_result_uses_canonical_snapshot_metadata(agent_harness) -> None:
    """Would fail if a tool returned caller labels instead of snapshot IDs, names, refs, and text."""
    from src.agent import AgentContext, search_documents

    arguments = json.dumps({"queries": ["fact"], "file_ids": ["old"], "limit": 1})
    value = json.loads(
        await search_documents.on_invoke_tool(
            _tool_context(AgentContext(await agent_harness.store.capture(), agent_harness.rag), "search_documents", arguments),
            arguments,
        )
    )

    assert value == {
        "results": [
            {"file_id": "old", "file_name": "old.pdf", "refs": ["p. 1"], "text": "Nội dung chuẩn"}
        ]
    }


@pytest.mark.asyncio
async def test_completed_sdk_stream_commits_only_after_nonempty_final_output(agent_harness, monkeypatch) -> None:
    """Would fail if a partial stream wrote durable items before final validation."""
    from src import agent

    observed: list[_CompletedStream] = []

    def run_streamed(*args: Any, **kwargs: Any) -> _CompletedStream:
        result = _CompletedStream(kwargs["session"])
        result.max_turns = kwargs["max_turns"]
        result.run_config = kwargs["run_config"]
        observed.append(result)
        return result

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    events = [event async for event in agent_harness.service.stream("s1", "question")]

    assert [event.type for event in events] == ["start", "delta", "done"]
    assert len(agent_harness.sessions.sdk_session("s1").items) == 2
    assert observed[0].max_turns == 4
    assert observed[0].run_config.tracing_disabled is True
    assert observed[0].run_config.session_settings.limit == 48
    assert agent_harness.sessions.sdk_session("s1").close_calls == 1


@pytest.mark.asyncio
async def test_empty_final_sdk_output_discards_the_turn(agent_harness, monkeypatch) -> None:
    """Would fail if an empty final answer became a complete durable conversation turn."""
    from src import agent

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"], answer=" "),
    )
    events = [event async for event in agent_harness.service.stream("s1", "question")]

    assert [event.type for event in events] == ["start", "delta", "error"]
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_cancelled_sdk_stream_discards_the_turn(agent_harness, monkeypatch) -> None:
    """Would fail if cancelling a stream committed its buffered user/assistant items."""
    from src import agent

    def run_streamed(*args: Any, **kwargs: Any) -> _CompletedStream:
        return _CompletedStream(kwargs["session"])

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    await agent_harness.service.stop("s1")
    assert [event.type async for event in stream] == ["cancelled"]
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_stop_all_discards_every_real_sdk_shaped_cancelled_turn(agent_harness, monkeypatch) -> None:
    """Would fail if SDK cancellation's is_complete flag were mistaken for success."""
    from src import agent

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"]),
    )
    streams = [agent_harness.service.stream(f"s{index}", "question") for index in range(2)]
    assert [await anext(stream) for stream in streams]
    await agent_harness.service.stop_all()

    assert [[event.type async for event in stream] for stream in streams] == [
        ["cancelled"],
        ["cancelled"],
    ]
    assert all(not agent_harness.sessions.sdk_session(f"s{index}").items for index in range(2))


@pytest.mark.asyncio
async def test_stop_all_skips_a_stale_identity_that_committed_before_cancellation(agent_harness, monkeypatch) -> None:
    """Would fail if a stop-all snapshot could cancel a run after its done event."""
    from src import agent

    results: list[_BarrierCompletedStream] = []

    def run_streamed(*args: Any, **kwargs: Any) -> _BarrierCompletedStream:
        result = _BarrierCompletedStream(kwargs["session"])
        results.append(result)
        return result

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    original_request = agent_harness.service._request_sdk_cancellation
    cancellation_captured = asyncio.Event()
    release_cancellation = asyncio.Event()

    async def delayed_request(*args: Any) -> Any:
        cancellation_captured.set()
        await release_cancellation.wait()
        return await original_request(*args)

    monkeypatch.setattr(agent_harness.service, "_request_sdk_cancellation", delayed_request)
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    finishing = asyncio.create_task(anext(stream))
    await results[0].waiting_to_finish.wait()
    stopping = asyncio.create_task(agent_harness.service.stop_all())
    await cancellation_captured.wait()

    results[0].allow_finish.set()
    assert (await finishing).type == "done"
    release_cancellation.set()
    await stopping

    assert results[0].cancel_calls == 0


@pytest.mark.asyncio
async def test_stop_suppresses_a_queued_late_delta(agent_harness, monkeypatch) -> None:
    """Would fail if queued SDK output were translated after service cancellation."""
    from src import agent

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _QueuedLateStream(kwargs["session"]),
    )
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    early = await anext(stream)
    assert (early.type, early.text) == ("delta", "early")

    await agent_harness.service.stop("s1")

    assert [event.type async for event in stream] == ["cancelled"]
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_repeated_cancellation_during_drain_settles_before_propagating(agent_harness, monkeypatch) -> None:
    """Would fail if a second cancellation released a running SDK cleanup early."""
    from src import agent

    agent_harness.service._run_gate = asyncio.Semaphore(1)
    results: list[_BackgroundStream] = []

    def run_streamed(*args: Any, **kwargs: Any) -> _BackgroundStream:
        result = _BackgroundStream(kwargs["session"])
        results.append(result)
        return result

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"
    active = agent_harness.service._active["s1"]
    closing = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(results[0].cancel_requested.wait(), timeout=1)
    closing.cancel()
    closing.cancel()

    assert not closing.done()
    assert agent_harness.service._run_gate.locked()
    assert "s1" in agent_harness.service._active
    results[0].allow_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=1)

    assert results[0].iterator_settled.is_set()
    assert not agent_harness.service._run_gate.locked()
    assert "s1" not in agent_harness.service._active
    assert active.settled.is_set()
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_repeated_cancellation_before_gate_settles_active_run(agent_harness, monkeypatch) -> None:
    """Would fail if cancellation while snapshot setup ran leaked the active session slot."""
    from src import agent

    capture_started = asyncio.Event()
    release_capture = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    original_capture = agent_harness.store.capture
    original_request = agent_harness.service._request_sdk_cancellation

    async def delayed_capture():
        capture_started.set()
        await release_capture.wait()
        return await original_capture()

    async def delayed_request(*args: Any) -> Any:
        cleanup_started.set()
        await release_cleanup.wait()
        return await original_request(*args)

    monkeypatch.setattr(agent_harness.store, "capture", delayed_capture)
    monkeypatch.setattr(agent_harness.service, "_request_sdk_cancellation", delayed_request)
    waiting = asyncio.create_task(anext(agent_harness.service.stream("s1", "question")))
    await capture_started.wait()
    active = agent_harness.service._active["s1"]
    waiting.cancel()
    await cleanup_started.wait()
    waiting.cancel()

    assert not waiting.done()
    assert "s1" in agent_harness.service._active
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiting, timeout=1)

    assert "s1" not in agent_harness.service._active
    assert active.settled.is_set()


@pytest.mark.asyncio
async def test_cancellation_after_commit_linearization_keeps_complete_turn(agent_harness, monkeypatch) -> None:
    """Would fail if cancelling an in-flight SDK session append left a partial turn."""
    from src import agent

    durable = _CommitBarrierSession()
    agent_harness.sessions.sessions["s1"] = durable
    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"]),
    )
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"
    completing = asyncio.create_task(anext(stream))
    await durable.commit_started.wait()
    completing.cancel()
    durable.allow_commit.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(completing, timeout=1)

    assert len(durable.items) == 2
    assert "s1" not in agent_harness.service._active


@pytest.mark.asyncio
async def test_cancellation_before_commit_linearization_discards_the_turn(agent_harness, monkeypatch) -> None:
    """Would fail if a stop admitted a commit after cancellation was already recorded."""
    from src import agent

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"]),
    )
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"
    await agent_harness.service.stop("s1")

    assert [event.type async for event in stream] == ["cancelled"]
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_stop_winning_the_commit_lock_discards_without_done(agent_harness, monkeypatch) -> None:
    """A stop queued between validation and commit linearization still wins the turn."""
    from src import agent

    class _CountingLock:
        def __init__(self) -> None:
            self._lock = asyncio.Lock()
            self.waiters = 0
            self.first_waiter = asyncio.Event()
            self.second_waiter = asyncio.Event()
            self.manual_task = asyncio.current_task()
            self.completion_task: asyncio.Task[Any] | None = None
            self.block_completion = False
            self.completion_requeued = asyncio.Event()
            self.allow_completion = asyncio.Event()

        async def acquire(self) -> bool:
            current = asyncio.current_task()
            if self.block_completion and current is self.completion_task:
                self.completion_requeued.set()
                await self.allow_completion.wait()
            if self._lock.locked():
                self.waiters += 1
                if self.waiters == 1:
                    self.first_waiter.set()
                if self.waiters == 2:
                    self.second_waiter.set()
            acquired = await self._lock.acquire()
            if (
                self.completion_task is None
                and current is not None
                and current is not self.manual_task
            ):
                self.completion_task = current
            return acquired

        def release(self) -> None:
            self._lock.release()
            if asyncio.current_task() is self.completion_task:
                self.block_completion = True

        async def __aenter__(self) -> "_CountingLock":
            await self.acquire()
            return self

        async def __aexit__(self, *args: Any) -> None:
            self.release()

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"]),
    )
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"

    lock = _CountingLock()
    agent_harness.service._active_lock = lock
    await lock.acquire()
    completing = asyncio.create_task(anext(stream))
    await lock.first_waiter.wait()
    stopping = asyncio.create_task(agent_harness.service.stop("s1"))
    await lock.second_waiter.wait()
    lock.release()
    await stopping
    await lock.completion_requeued.wait()
    lock.allow_completion.set()

    assert (await completing).type == "cancelled"
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_caller_cancellation_keeps_the_sdk_event_iterator_open_for_drain(agent_harness, monkeypatch) -> None:
    """Would fail if caller cancellation directly owned SDK iterator __anext__()."""
    from src import agent

    agent_harness.service._run_gate = asyncio.Semaphore(1)
    results: list[_OwnerCancelledStream] = []

    def run_streamed(*args: Any, **kwargs: Any) -> _OwnerCancelledStream:
        result = _OwnerCancelledStream(kwargs["session"])
        results.append(result)
        return result

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"
    active = agent_harness.service._active["s1"]
    waiting = asyncio.create_task(anext(stream))
    await results[0].events.next_started.wait()
    waiting.cancel()
    await results[0].cancel_requested.wait()

    assert not results[0].events.owner_cancelled
    assert not waiting.done()
    assert agent_harness.service._run_gate.locked()
    assert "s1" in agent_harness.service._active
    results[0].allow_settlement.set()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert results[0].events.iterator_settled.is_set()
    assert results[0].run_loop_settled.is_set()
    assert not agent_harness.service._run_gate.locked()
    assert "s1" not in agent_harness.service._active
    assert active.settled.is_set()
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_snapshot_capture_failure_yields_one_error_and_cleans_active(agent_harness, monkeypatch) -> None:
    """Setup failures must not turn into an empty stream response."""

    async def fail_capture() -> IndexSnapshot:
        raise RuntimeError("snapshot unavailable")

    monkeypatch.setattr(agent_harness.store, "capture", fail_capture)

    assert [event.type async for event in agent_harness.service.stream("s1", "question")] == [
        "error"
    ]
    assert "s1" not in agent_harness.service._active


@pytest.mark.asyncio
async def test_disconnect_discards_a_partially_streamed_turn(agent_harness, monkeypatch) -> None:
    """Would fail if closing an SSE consumer left the SDK background run alive."""
    from src import agent

    agent_harness.service._run_gate = asyncio.Semaphore(1)
    results: list[_BackgroundStream] = []

    def run_streamed(*args: Any, **kwargs: Any) -> _BackgroundStream:
        result = _BackgroundStream(kwargs["session"])
        results.append(result)
        return result

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    stream = agent_harness.service.stream("s1", "question")
    events = [await anext(stream), await anext(stream)]
    try:
        await results[0].background_started.wait()
        closing = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(results[0].cancel_requested.wait(), timeout=1)

        assert [event.type for event in events] == ["start", "delta"]
        assert results[0].cancel_calls == 1
        assert not closing.done()
        assert agent_harness.service._run_gate.locked()
        assert "s1" in agent_harness.service._active

        results[0].allow_settlement.set()
        await asyncio.wait_for(closing, timeout=1)
        assert results[0].iterator_settled.is_set()
        assert results[0]._background_task.cancelled()
        assert not agent_harness.service._run_gate.locked()
        assert "s1" not in agent_harness.service._active
        assert agent_harness.sessions.sdk_session("s1").items == []

        next_stream = agent_harness.service.stream("s2", "question")
        assert (await asyncio.wait_for(anext(next_stream), timeout=1)).type == "start"
        results[1].allow_settlement.set()
        await next_stream.aclose()
    finally:
        for result in results:
            result.cancel()
            result.allow_settlement.set()
        await asyncio.gather(
            *(result._background_task for result in results), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_stop_during_snapshot_capture_cancels_the_new_sdk_stream(agent_harness, monkeypatch) -> None:
    """Would fail if a stop racing setup were lost before Runner receives the stream."""
    from src import agent

    captured = asyncio.Event()
    release = asyncio.Event()
    original_capture = agent_harness.store.capture

    async def delayed_capture():
        captured.set()
        await release.wait()
        return await original_capture()

    monkeypatch.setattr(agent_harness.store, "capture", delayed_capture)
    monkeypatch.setattr(agent.Runner, "run_streamed", lambda *args, **kwargs: _CompletedStream(kwargs["session"]))
    stream = agent_harness.service.stream("s1", "question")
    start = asyncio.create_task(anext(stream))
    await captured.wait()
    await agent_harness.service.stop("s1")
    release.set()
    assert (await start).type == "start"
    assert [event.type async for event in stream] == ["cancelled"]
    assert agent_harness.sessions.sdk_session("s1").items == []


@pytest.mark.asyncio
async def test_four_sessions_can_enter_runner_while_duplicates_are_rejected(agent_harness, monkeypatch) -> None:
    """Would fail if an application-wide lock serialized independent sessions."""
    from src import agent

    entered = asyncio.Event()
    release = asyncio.Event()
    running = 0
    maximum = 0

    class BlockingStream(_CompletedStream):
        async def stream_events(self):
            nonlocal running, maximum
            running += 1
            maximum = max(maximum, running)
            if running == 4:
                entered.set()
            await release.wait()
            running -= 1
            self.is_complete = True
            self.final_output = "ok"
            if False:
                yield None

    def run_streamed(*args: Any, **kwargs: Any) -> BlockingStream:
        return BlockingStream(kwargs["session"])

    monkeypatch.setattr(agent.Runner, "run_streamed", run_streamed)
    streams = [agent_harness.service.stream(f"s{index}", "question") for index in range(4)]
    starters = [asyncio.create_task(anext(stream)) for stream in streams]
    await asyncio.gather(*starters)
    finishers = [asyncio.create_task(anext(stream)) for stream in streams]
    with pytest.raises(ValueError, match="already active"):
        await anext(agent_harness.service.stream("s0", "again"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    release.set()
    await asyncio.gather(*finishers)
    assert maximum == 4
