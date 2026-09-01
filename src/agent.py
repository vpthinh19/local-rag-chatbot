"""One Agents SDK document agent and a small application-event translation layer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Annotated, Any, Literal, Protocol

from agents import Agent, ModelSettings, RunConfig, RunContextWrapper, Runner, function_tool
from agents.memory import SessionSettings
from pydantic import Field, StringConstraints

from src.config import Settings
from src.rag import IndexSnapshot, RagService, SnapshotStore
from src.sessions import TransactionalSession, bounded_session_input


_TOOL_RESULT_CHAR_LIMIT = 48_000
_Query = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_FileId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_Queries = Annotated[list[_Query], Field(min_length=1, max_length=3)]
_FileIds = Annotated[list[_FileId], Field(min_length=1, max_length=8)]
_Limit = Annotated[int, Field(ge=1, le=6)]


@dataclass(frozen=True, slots=True)
class AgentContext:
    """The immutable retrieval view captured for exactly one SDK run."""

    snapshot: IndexSnapshot
    rag: RagService


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A stable, content-safe application event for a chat stream."""

    type: Literal["start", "status", "delta", "done", "error", "cancelled"]
    text: str = ""


class _SessionFactory(Protocol):
    def sdk_session(self, session_id: str) -> Any: ...


def _compact_json(value: object) -> str:
    """Serialize a whole compact payload or fail before returning partial JSON."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) > _TOOL_RESULT_CHAR_LIMIT:
        raise ValueError("tool result is too large")
    return encoded


def _documents_for(context: AgentContext, file_ids: list[str]) -> list[Any]:
    """Resolve only canonical ready IDs in the run's already-captured snapshot."""
    by_id = {document.id: document for document in context.snapshot.documents}
    resolved: list[Any] = []
    seen: set[str] = set()
    for file_id in file_ids:
        document = by_id.get(file_id)
        if document is None or document.status != "ready":
            raise ValueError("requested document is not available")
        if file_id not in seen:
            resolved.append(document)
            seen.add(file_id)
    return resolved


@function_tool(failure_error_function=None)
async def get_document_overviews(
    context: RunContextWrapper[AgentContext], file_ids: _FileIds
) -> str:
    """Lấy overview để tóm tắt, lập dàn ý, hoặc so sánh khái quát tài liệu."""
    documents = _documents_for(context.context, file_ids)
    return _compact_json(
        {
            "documents": [
                {
                    "file_id": document.id,
                    "file_name": document.file_name,
                    "overview": document.overview,
                }
                for document in documents
            ]
        }
    )


@function_tool(failure_error_function=None)
async def search_documents(
    context: RunContextWrapper[AgentContext],
    queries: _Queries,
    file_ids: _FileIds,
    limit: _Limit,
) -> str:
    """Tìm đoạn trích cho câu hỏi chi tiết hoặc dữ kiện cụ thể trong tài liệu."""
    _documents_for(context.context, file_ids)
    chunks = await context.context.rag.search(
        context.context.snapshot, list(queries), list(file_ids), limit
    )
    return _compact_json(
        {
            "results": [
                {
                    "file_id": chunk.document_id,
                    "file_name": next(
                        document.file_name
                        for document in context.context.snapshot.documents
                        if document.id == chunk.document_id
                    ),
                    "refs": list(chunk.refs),
                    "text": chunk.text,
                }
                for chunk in chunks
            ]
        }
    )


def dynamic_instructions(context: RunContextWrapper[AgentContext], _: Agent[AgentContext]) -> str:
    """Give this run its immutable ready-document catalogue and grounding rules."""
    documents = [
        {"file_id": document.id, "file_name": document.file_name}
        for document in context.context.snapshot.documents
        if document.status == "ready"
    ]
    return (
        "Bạn là trợ lý tài liệu thân thiện. Trả lời trực tiếp lời chào và hội thoại "
        "thông thường. Khi cần nội dung tài liệu, dùng get_document_overviews cho "
        "tóm tắt/dàn ý/so sánh khái quát, hoặc search_documents cho dữ kiện cụ thể. "
        "Chỉ khẳng định dữ kiện tài liệu có trong kết quả công cụ; sau khi search, "
        "trích dẫn tên file và refs được cung cấp. Nếu không có kết quả, nói không "
        "tìm thấy và không suy đoán. Không có công cụ ghi, xóa, hoặc sửa tài liệu. "
        "Tài liệu sẵn sàng trong lượt này: "
        + json.dumps(documents, ensure_ascii=False, separators=(",", ":"))
    )


def build_document_agent(responses_model: Any) -> Agent[AgentContext]:
    """Build the sole immutable application agent once during runtime composition."""
    return Agent[AgentContext](
        name="Local document agent",
        instructions=dynamic_instructions,
        model=responses_model,
        model_settings=ModelSettings(temperature=0.1, parallel_tool_calls=False),
        tools=[search_documents, get_document_overviews],
    )


@dataclass(slots=True)
class _ActiveRun:
    settled: asyncio.Event
    stream: Any | None = None
    cancellation_requested: bool = False
    sdk_cancel_requested: bool = False
    completed: bool = False


class AgentService:
    """Run one SDK stream per session and expose only application-level events."""

    def __init__(
        self,
        settings: Settings,
        snapshots: SnapshotStore,
        rag: RagService,
        sessions: _SessionFactory,
        *,
        responses_model: Any,
    ) -> None:
        self._settings = settings
        self._snapshots = snapshots
        self._rag = rag
        self._sessions = sessions
        self._agent = build_document_agent(responses_model)
        self._run_gate = asyncio.Semaphore(settings.llm_concurrency)
        self._active: dict[str, _ActiveRun] = {}
        self._active_lock = asyncio.Lock()

    @property
    def agent(self) -> Agent[AgentContext]:
        """Expose the one shared agent for runtime diagnostics and tests."""
        return self._agent

    async def stream(self, session_id: str, message: str) -> AsyncIterator[AgentEvent]:
        """Stream one transactionally persisted SDK run for an independent session."""
        user_message = self._validate_message(message)
        active = _ActiveRun(asyncio.Event())
        async with self._active_lock:
            if session_id in self._active:
                raise ValueError("a chat run is already active for this session")
            self._active[session_id] = active

        transaction: TransactionalSession | None = None
        committed = False
        cancelled = False
        inner_cleanup_finished = False
        event_iterator: AsyncIterator[object] | None = None
        try:
            snapshot = await self._snapshots.capture()
            transaction = TransactionalSession(self._sessions.sdk_session(session_id))
            context = AgentContext(snapshot, self._rag)
            async with self._run_gate:
                try:
                    result = Runner.run_streamed(
                        self._agent,
                        user_message,
                        context=context,
                        session=transaction,
                        max_turns=self._settings.agent_max_turns,
                        run_config=RunConfig(
                            tracing_disabled=True,
                            session_settings=SessionSettings(limit=self._settings.session_raw_item_limit),
                            session_input_callback=lambda history, new: bounded_session_input(
                                history,
                                new,
                                max_messages=self._settings.session_visible_message_limit,
                                max_chars=self._settings.session_context_chars,
                            ),
                        ),
                    )
                    async with self._active_lock:
                        active.stream = result
                        stream_to_cancel = (
                            result
                            if active.cancellation_requested
                            and not active.sdk_cancel_requested
                            and not bool(getattr(result, "is_complete", False))
                            else None
                        )
                        if stream_to_cancel is not None:
                            active.sdk_cancel_requested = True
                    if stream_to_cancel is not None:
                        stream_to_cancel.cancel()
                    event_iterator = result.stream_events()
                    yield AgentEvent("start")
                    async for event in event_iterator:
                        translated = self._translate(event)
                        if translated is not None:
                            yield translated
                    async with self._active_lock:
                        cancelled = active.cancellation_requested or not bool(
                            getattr(result, "is_complete", False)
                        )
                    failure = getattr(result, "run_loop_exception", None)
                    if failure is not None:
                        raise failure
                    if cancelled:
                        yield AgentEvent("cancelled")
                        return
                    answer = getattr(result, "final_output", None)
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("agent returned an empty final answer")
                    async with self._active_lock:
                        if active.cancellation_requested:
                            cancelled = True
                        else:
                            await transaction.commit()
                            committed = True
                            active.completed = True
                            if self._active.get(session_id) is active:
                                del self._active[session_id]
                    if cancelled:
                        yield AgentEvent("cancelled")
                        return
                    yield AgentEvent("done")
                except Exception:
                    async with self._active_lock:
                        cancelled = active.cancellation_requested
                    raise
                finally:
                    if not committed:
                        await self._cleanup_uncommitted(
                            session_id, active, transaction, event_iterator
                        )
                        inner_cleanup_finished = True
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception:
            yield AgentEvent("cancelled" if cancelled else "error")
        finally:
            if not committed and not inner_cleanup_finished:
                await self._cleanup_uncommitted(
                    session_id, active, transaction, event_iterator
                )
            active.settled.set()
            async with self._active_lock:
                if self._active.get(session_id) is active:
                    del self._active[session_id]

    async def stop(self, session_id: str) -> None:
        """Request safe cancellation of the exact active session stream, if any."""
        async with self._active_lock:
            active = self._active.get(session_id)
        stream = (
            await self._request_sdk_cancellation(session_id, active)
            if active is not None
            else None
        )
        if stream is not None:
            stream.cancel()

    async def stop_all(self) -> None:
        """Request cancellation for every active stream without coupling sessions."""
        async with self._active_lock:
            active_runs = tuple(self._active.items())
        streams = await asyncio.gather(
            *(
                self._request_sdk_cancellation(session_id, active)
                for session_id, active in active_runs
            )
        )
        for stream in streams:
            if stream is not None:
                stream.cancel()

    async def _request_sdk_cancellation(
        self, session_id: str, active: _ActiveRun
    ) -> Any | None:
        """Record cancellation before requesting the matching SDK stream to stop."""
        async with self._active_lock:
            if self._active.get(session_id) is not active or active.completed:
                return None
            active.cancellation_requested = True
            if (
                active.stream is None
                or active.sdk_cancel_requested
                or bool(getattr(active.stream, "is_complete", False))
            ):
                return None
            active.sdk_cancel_requested = True
            return active.stream

    async def _cleanup_uncommitted(
        self,
        session_id: str,
        active: _ActiveRun,
        transaction: TransactionalSession | None,
        event_iterator: AsyncIterator[object] | None,
    ) -> None:
        """Cancel and drain one SDK run before releasing its application run slot."""
        stream = await self._request_sdk_cancellation(session_id, active)
        if stream is not None:
            try:
                stream.cancel()
            except Exception:
                pass
        if event_iterator is not None:
            try:
                async for _ in event_iterator:
                    pass
            except (Exception, asyncio.CancelledError):
                pass
        if transaction is not None:
            try:
                await transaction.discard()
            except (Exception, asyncio.CancelledError):
                pass

    def _validate_message(self, message: object) -> str:
        if not isinstance(message, str) or not (clean := message.strip()):
            raise ValueError("chat message must not be empty")
        if len(clean) > self._settings.max_message_chars:
            raise ValueError("chat message exceeds the size limit")
        return clean

    @staticmethod
    def _translate(event: object) -> AgentEvent | None:
        if getattr(event, "type", None) == "raw_response_event":
            data = getattr(event, "data", None)
            if getattr(data, "type", None) == "response.output_text.delta":
                delta = getattr(data, "delta", "")
                if isinstance(delta, str) and delta:
                    return AgentEvent("delta", delta)
        if getattr(event, "type", None) == "run_item_stream_event" and getattr(event, "name", None) == "tool_called":
            return AgentEvent("status", "Đang tra cứu tài liệu.")
        return None
