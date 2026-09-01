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

    async def get_items(self, limit: int | None = None) -> list[Any]:
        return self.items if limit is None else self.items[-limit:]

    async def add_items(self, items: list[Any]) -> None:
        self.items.extend(items)

    async def pop_item(self) -> Any | None:
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        self.items.clear()


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
async def test_disconnect_discards_a_partially_streamed_turn(agent_harness, monkeypatch) -> None:
    """Would fail if closing an SSE consumer committed its already-buffered SDK items."""
    from src import agent

    monkeypatch.setattr(
        agent.Runner,
        "run_streamed",
        lambda *args, **kwargs: _CompletedStream(kwargs["session"]),
    )
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    assert (await anext(stream)).type == "delta"
    await stream.aclose()

    assert agent_harness.sessions.sdk_session("s1").items == []


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
