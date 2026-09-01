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
    commit_started: bool = False
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
        event_iterator: AsyncIterator[object] | None = None
        cleanup_finished = False
        cancellation_deferred = False
        error_event = False
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
                    while True:
                        pending_event = asyncio.create_task(anext(event_iterator))
                        try:
                            event = await asyncio.shield(pending_event)
                        except StopAsyncIteration:
                            break
                        except asyncio.CancelledError:
                            cancellation_deferred = (
                                await self._settle_pending_event_shielded(
                                    session_id, active, pending_event
                                )
                                or cancellation_deferred
                            )
                            raise
                        if await self._cancellation_recorded(session_id, active):
                            yield AgentEvent("cancelled")
                            return
                        translated = self._translate(event)
                        if translated is not None:
                            yield translated
                    failure = getattr(result, "run_loop_exception", None)
                    if failure is not None:
                        raise failure
                    if (
                        await self._cancellation_recorded(session_id, active)
                        or not bool(getattr(result, "is_complete", False))
                    ):
                        yield AgentEvent("cancelled")
                        return
                    answer = getattr(result, "final_output", None)
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("agent returned an empty final answer")
                    commit_cancelled, commit_cancellation_deferred = await self._commit_complete(
                        session_id, active, transaction
                    )
                    cancellation_deferred = (
                        cancellation_deferred or commit_cancellation_deferred
                    )
                    if commit_cancelled:
                        yield AgentEvent("cancelled")
                        return
                    if not cancellation_deferred:
                        yield AgentEvent("done")
                except Exception:
                    error_event = not await self._cancellation_recorded(session_id, active)
                    raise
                finally:
                    if not active.completed:
                        inner_cancellation_deferred = (
                            await self._finalize_uncommitted_shielded(
                                session_id, active, transaction, event_iterator
                            )
                        )
                        cancellation_deferred = (
                            inner_cancellation_deferred or cancellation_deferred
                        )
                        cleanup_finished = True
                        if inner_cancellation_deferred:
                            raise asyncio.CancelledError
        except asyncio.CancelledError:
            cancellation_deferred = True
        except Exception:
            error_event = error_event or not await self._cancellation_recorded(
                session_id, active
            )
        finally:
            if not active.completed and not cleanup_finished:
                cancellation_deferred = (
                    await self._finalize_uncommitted_shielded(
                        session_id, active, transaction, event_iterator
                    )
                    or cancellation_deferred
                )
                cleanup_finished = True
            if active.completed:
                cancellation_deferred = (
                    await self._finish_completed_shielded(session_id, active)
                    or cancellation_deferred
                )
        if cancellation_deferred:
            raise asyncio.CancelledError
        if error_event:
            yield AgentEvent("error")

    async def stop(self, session_id: str) -> None:
        """Request safe cancellation of the exact active session stream, if any."""
        async with self._active_lock:
            active = self._active.get(session_id)
        if active is not None:
            await self._request_and_cancel_sdk_run(session_id, active)

    async def stop_and_settle(self, session_id: str) -> None:
        """Cancel one session and wait until its SDK turn is no longer active."""
        async with self._active_lock:
            active = self._active.get(session_id)
        if active is None:
            return
        await self._request_and_cancel_sdk_run(session_id, active)
        await active.settled.wait()

    async def stop_all(self) -> None:
        """Request cancellation for every active stream without coupling sessions."""
        async with self._active_lock:
            active_runs = tuple(self._active.items())
        await asyncio.gather(
            *(
                self._request_and_cancel_sdk_run(session_id, active)
                for session_id, active in active_runs
            )
        )

    async def stop_all_and_settle(self) -> None:
        """Cancel all streams and await the settlement of that exact run snapshot."""
        async with self._active_lock:
            active_runs = tuple(self._active.items())
        await asyncio.gather(
            *(self._request_and_cancel_sdk_run(session_id, active) for session_id, active in active_runs)
        )
        await asyncio.gather(*(active.settled.wait() for _, active in active_runs))

    async def _request_sdk_cancellation(
        self, session_id: str, active: _ActiveRun
    ) -> Any | None:
        """Record cancellation before requesting the matching SDK stream to stop."""
        async with self._active_lock:
            if (
                self._active.get(session_id) is not active
                or active.completed
                or active.commit_started
            ):
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

    async def _cancellation_recorded(self, session_id: str, active: _ActiveRun) -> bool:
        """Read cancellation only while this exact active run remains cancellable."""
        async with self._active_lock:
            return (
                self._active.get(session_id) is active
                and active.cancellation_requested
                and not active.commit_started
            )

    async def _commit_complete(
        self,
        session_id: str,
        active: _ActiveRun,
        transaction: TransactionalSession,
    ) -> tuple[bool, bool]:
        """Cross the durable-turn linearization point and finish its single append."""
        async with self._active_lock:
            if (
                self._active.get(session_id) is not active
                or active.cancellation_requested
            ):
                return True, False
            active.commit_started = True
        commit_task = asyncio.create_task(transaction.commit())
        _result, cancellation_deferred = await self._await_shielded(commit_task)
        mark_task = asyncio.create_task(self._mark_completed(active))
        _result, mark_cancellation_deferred = await self._await_shielded(mark_task)
        cancellation_deferred = cancellation_deferred or mark_cancellation_deferred
        cancellation_deferred = (
            await self._finish_completed_shielded(session_id, active)
            or cancellation_deferred
        )
        return False, cancellation_deferred

    async def _mark_completed(self, active: _ActiveRun) -> None:
        """Record the known-successful durable append before any cancellation can unwind."""
        async with self._active_lock:
            active.completed = True

    async def _finalize_uncommitted_shielded(
        self,
        session_id: str,
        active: _ActiveRun,
        transaction: TransactionalSession | None,
        event_iterator: AsyncIterator[object] | None,
    ) -> bool:
        """Settle uncommitted SDK work despite repeated caller cancellation."""
        task = asyncio.create_task(
            self._finalize_uncommitted(session_id, active, transaction, event_iterator)
        )
        _result, cancellation_deferred = await self._await_shielded(task)
        return cancellation_deferred

    async def _settle_pending_event_shielded(
        self,
        session_id: str,
        active: _ActiveRun,
        pending_event: asyncio.Task[object],
    ) -> bool:
        """Keep caller cancellation from closing the SDK event iterator mid-next."""
        task = asyncio.create_task(
            self._cancel_and_settle_pending_event(session_id, active, pending_event)
        )
        _result, cancellation_deferred = await self._await_shielded(task)
        return cancellation_deferred

    async def _cancel_and_settle_pending_event(
        self,
        session_id: str,
        active: _ActiveRun,
        pending_event: asyncio.Task[object],
    ) -> None:
        await self._request_and_cancel_sdk_run(session_id, active)
        try:
            await pending_event
        except (StopAsyncIteration, asyncio.CancelledError, Exception):
            pass

    async def _finalize_uncommitted(
        self,
        session_id: str,
        active: _ActiveRun,
        transaction: TransactionalSession | None,
        event_iterator: AsyncIterator[object] | None,
    ) -> None:
        await self._cleanup_uncommitted(session_id, active, transaction, event_iterator)
        await self._remove_active(session_id, active)

    async def _finish_completed_shielded(
        self, session_id: str, active: _ActiveRun
    ) -> bool:
        """Publish a completed durable turn before deferred cancellation propagates."""
        task = asyncio.create_task(self._remove_active(session_id, active))
        _result, cancellation_deferred = await self._await_shielded(task)
        return cancellation_deferred

    async def _remove_active(self, session_id: str, active: _ActiveRun) -> None:
        """Release the exact active-session record after its terminal outcome is known."""
        async with self._active_lock:
            if self._active.get(session_id) is active:
                del self._active[session_id]
        active.settled.set()

    @staticmethod
    async def _await_shielded(task: asyncio.Task[Any]) -> tuple[Any, bool]:
        """Await a terminal task through repeated cancellation, deferring propagation."""
        cancellation_deferred = False
        while True:
            try:
                return await asyncio.shield(task), cancellation_deferred
            except asyncio.CancelledError:
                if task.done():
                    raise
                cancellation_deferred = True
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()

    async def _cleanup_uncommitted(
        self,
        session_id: str,
        active: _ActiveRun,
        transaction: TransactionalSession | None,
        event_iterator: AsyncIterator[object] | None,
    ) -> None:
        """Cancel and drain one SDK run before releasing its application run slot."""
        await self._request_and_cancel_sdk_run(session_id, active)
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

    async def _request_and_cancel_sdk_run(
        self, session_id: str, active: _ActiveRun
    ) -> None:
        """Record cancellation and make the one permitted SDK cancellation request."""
        stream = await self._request_sdk_cancellation(session_id, active)
        if stream is not None:
            try:
                stream.cancel()
            except Exception:
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
