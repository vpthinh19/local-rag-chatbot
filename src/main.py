"""FastAPI composition root and thin session/document routes."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx
import numpy as np
from pydantic import BaseModel

from src.agent import AgentEvent, AgentService
from src.config import Settings, settings as default_settings
from src.database import Database
from src.documents import DocumentService
from src.jobs import DocumentWorker
from src.migration import migrate_legacy
from src.model_clients import LocalModelClients, build_agent_model
from src.models import DataValidationError, DocumentRecord, SessionRecord
from src.rag import IndexSnapshot, RagService, SnapshotStore
from src.sessions import SessionService


@dataclass(slots=True)
class ApplicationRuntime:
    """The explicitly composed, application-owned runtime dependencies."""

    settings: Settings
    http: httpx.AsyncClient
    database: Database
    executor: ThreadPoolExecutor
    models: LocalModelClients
    rag: RagService
    snapshots: SnapshotStore
    documents: DocumentService
    parser: object
    sessions: SessionService
    worker: DocumentWorker
    agent: AgentService


class _RenameRequest(BaseModel):
    title: str


class _ChatRequest(BaseModel):
    message: str


def _record(record: DocumentRecord | SessionRecord) -> dict[str, object]:
    return asdict(record)


def _sse(value: dict[str, object]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _event_data(event: AgentEvent) -> dict[str, object]:
    value: dict[str, object] = {"type": event.type}
    if event.text:
        value["text"] = event.text
    return value


def _http_error(exc: DataValidationError, *, conflict: bool = False) -> HTTPException:
    detail = str(exc)
    if detail.endswith("does not exist"):
        return HTTPException(status_code=404, detail=detail)
    return HTTPException(status_code=409 if conflict else 400, detail=detail)


def create_app(
    app_settings: Settings | None = None,
    *,
    model_transport: httpx.AsyncBaseTransport | None = None,
    heartbeat_interval: float = 10.0,
) -> FastAPI:
    """Build the application without embedding domain orchestration in routes."""
    configured = app_settings or default_settings
    static_dir = Path(__file__).parent / "static"
    template_path = Path(__file__).parent / "templates" / "index.html"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configured.ensure_dirs()
        timeout = httpx.Timeout(
            connect=configured.http_connect_timeout,
            read=configured.http_read_timeout,
            write=configured.http_write_timeout,
            pool=configured.http_pool_timeout,
        )
        http = httpx.AsyncClient(timeout=timeout, transport=model_transport)
        executor = ThreadPoolExecutor(max_workers=configured.rag_cpu_workers)
        try:
            database = Database(configured.database_path, configured.database_busy_timeout_ms)
            await database.initialize()
            sessions = SessionService(configured, database)
            await migrate_legacy(configured, database, sessions.sdk_session)
            models = LocalModelClients(
                configured,
                http,
                embedding_gate=asyncio.Semaphore(configured.embedding_concurrency),
                rerank_gate=asyncio.Semaphore(configured.rerank_concurrency),
            )
            rag = RagService(
                models,
                cpu_executor=executor,
                embedding_batch_size=configured.embedding_batch_size,
                lexical_candidate_limit=configured.lexical_candidate_limit,
                semantic_candidate_limit=configured.semantic_candidate_limit,
                fused_candidate_limit=configured.fused_candidate_limit,
                final_chunk_limit=configured.final_chunk_limit,
            )
            snapshots = SnapshotStore(
                IndexSnapshot((), (), np.empty((0, 0), dtype=np.float32), None)
            )
            documents = DocumentService(configured, database, snapshots.publication_lock)
            await documents.reconcile_files()
            worker = DocumentWorker(
                configured, database, documents, documents.parser, models, rag, snapshots
            )
            initial = await worker.build_ready_snapshot()
            async with snapshots.publication_lock:
                snapshots.install_locked(initial)
            await worker.recover()
            agent = AgentService(
                configured,
                snapshots,
                rag,
                sessions,
                responses_model=build_agent_model(configured, http),
            )
            runtime = ApplicationRuntime(
                configured, http, database, executor, models, rag, snapshots,
                documents, documents.parser, sessions, worker, agent,
            )
            app.state.runtime = runtime
            worker.start()
            yield
        finally:
            runtime = getattr(app.state, "runtime", None)
            if runtime is not None:
                await runtime.agent.stop_all_and_settle()
                await runtime.worker.stop()
                await runtime.parser.cancel_active()
            await http.aclose()
            executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Local RAG Chatbot", version="3.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(template_path)

    @app.post("/api/sessions", status_code=201)
    async def create_session() -> JSONResponse:
        return JSONResponse(_record(await _runtime(app).sessions.create()), status_code=201)

    @app.get("/api/sessions")
    async def list_sessions() -> JSONResponse:
        return JSONResponse({"sessions": [_record(item) for item in await _runtime(app).sessions.list()]})

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, body: _RenameRequest) -> JSONResponse:
        try:
            session = await _runtime(app).sessions.rename(session_id, body.title)
        except DataValidationError as exc:
            raise _http_error(exc) from exc
        return JSONResponse(_record(session))

    @app.get("/api/sessions/{session_id}/messages")
    async def session_messages(session_id: str) -> JSONResponse:
        try:
            messages = await _runtime(app).sessions.messages(session_id)
        except DataValidationError as exc:
            raise _http_error(exc) from exc
        return JSONResponse({"messages": [asdict(item) for item in messages]})

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        runtime = _runtime(app)
        if await runtime.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session does not exist")
        await runtime.agent.stop_and_settle(session_id)
        await runtime.sessions.delete(session_id)

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, body: _ChatRequest, request: Request) -> StreamingResponse:
        runtime = _runtime(app)
        if await runtime.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session does not exist")
        message = body.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="chat message must not be empty")
        if len(message) > runtime.settings.max_message_chars:
            raise HTTPException(status_code=413, detail="chat message exceeds the size limit")
        stream = runtime.agent.stream(session_id, message)
        try:
            first = await anext(stream)
        except ValueError as exc:
            if "already active" in str(exc):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await runtime.sessions.touch_from_first_message(session_id, message)

        async def events() -> AsyncIterator[str]:
            pending: asyncio.Task[AgentEvent] | None = None
            try:
                yield _sse(_event_data(first))
                pending = asyncio.create_task(anext(stream))
                while True:
                    if await request.is_disconnected():
                        return
                    try:
                        event = await asyncio.wait_for(asyncio.shield(pending), heartbeat_interval)
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    except StopAsyncIteration:
                        return
                    yield _sse(_event_data(event))
                    pending = asyncio.create_task(anext(stream))
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                await runtime.agent.stop_and_settle(session_id)
                await stream.aclose()

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache, no-store", "Connection": "keep-alive", "X-Accel-Buffering": "no",
            }
        )

    @app.post("/api/sessions/{session_id}/stop")
    async def stop_session(session_id: str) -> JSONResponse:
        runtime = _runtime(app)
        if await runtime.sessions.get(session_id) is None:
            raise HTTPException(status_code=404, detail="session does not exist")
        await runtime.agent.stop(session_id)
        return JSONResponse({"status": "ok"})

    @app.post("/api/documents", status_code=202)
    async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
        try:
            document = await _runtime(app).documents.create_upload(file)
        except DataValidationError as exc:
            raise _http_error(exc) from exc
        finally:
            await file.close()
        return JSONResponse(_record(document), status_code=202)

    @app.get("/api/documents")
    async def list_documents() -> JSONResponse:
        return JSONResponse({"documents": [_record(item) for item in await _runtime(app).documents.list()]})

    @app.get("/api/documents/{document_id}/download")
    async def download_document(document_id: str) -> FileResponse:
        documents = _runtime(app).documents
        document = await documents.get(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="document does not exist")
        try:
            path = await documents.download_path(document_id)
        except DataValidationError as exc:
            raise _http_error(exc, conflict=document.status == "deleting") from exc
        return FileResponse(path, filename=document.file_name, media_type=document.media_type)

    @app.post("/api/documents/{document_id}/retry", status_code=202)
    async def retry_document(document_id: str) -> JSONResponse:
        try:
            document = await _runtime(app).documents.retry(document_id)
        except DataValidationError as exc:
            raise _http_error(exc, conflict=True) from exc
        return JSONResponse(_record(document), status_code=202)

    @app.delete("/api/documents/{document_id}", status_code=202)
    async def delete_document(document_id: str) -> JSONResponse:
        try:
            document = await _runtime(app).documents.schedule_delete(document_id)
        except DataValidationError as exc:
            raise _http_error(exc) from exc
        return JSONResponse(_record(document), status_code=202)

    return app


def _runtime(app: FastAPI) -> ApplicationRuntime:
    runtime: ApplicationRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Application is not ready")
    return runtime


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, workers=1)


if __name__ == "__main__":
    run()
