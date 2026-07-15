"""FastAPI entrypoint for the compact local RAG agent."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import httpx

from src.chat import ChatAgent, LiveHistory
from src.config import Settings, settings as default_settings
from src.documents import DocumentService, LiveCorpus, RequestState
from src.llama import LlamaClient
from src.models import Corpus, History
from src.rag import RagIndex


@dataclass(slots=True)
class ApplicationRuntime:
    settings: Settings
    http: httpx.AsyncClient
    live_corpus: LiveCorpus
    live_history: LiveHistory
    rag: RagIndex
    documents: DocumentService
    chat: ChatAgent
    active: RequestState | None = None
    active_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def claim_request(self) -> RequestState | None:
        async with self.active_lock:
            if self.active is not None:
                return None
            state = RequestState(uuid4().hex)
            self.active = state
            return state

    async def release_request(self, state: RequestState) -> None:
        async with self.active_lock:
            if self.active is state:
                self.active = None

    async def has_active_request(self) -> bool:
        async with self.active_lock:
            return self.active is not None

    async def cancel_active(self) -> bool:
        async with self.active_lock:
            state = self.active
            if state is None:
                return False
            state.cancel_event.set()
            task = state.task
            if task is not None and not task.done():
                task.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        return True


def create_app(
    app_settings: Settings | None = None,
    *,
    model_transport: httpx.AsyncBaseTransport | None = None,
    heartbeat_interval: float = 10.0,
) -> FastAPI:
    configured = app_settings or default_settings
    static_dir = Path(__file__).parent / "static"
    template_path = Path(__file__).parent / "templates" / "index.html"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured.ensure_dirs()
        timeout = httpx.Timeout(
            connect=configured.http_connect_timeout,
            read=configured.http_read_timeout,
            write=configured.http_write_timeout,
            pool=configured.http_pool_timeout,
        )
        http = httpx.AsyncClient(timeout=timeout, transport=model_transport)
        try:
            llama = LlamaClient(
                http,
                configured.llm_url,
                configured.embed_url,
                configured.rerank_url,
            )
            rag = RagIndex(
                llama,
                batch_size=configured.embedding_batch_size,
                lexical_limit=configured.lexical_candidate_limit,
                semantic_limit=configured.semantic_candidate_limit,
                candidate_limit=configured.fused_candidate_limit,
                final_limit=configured.final_chunk_limit,
            )
            live_corpus = LiveCorpus(Corpus.load(configured.corpus_path))
            live_history = LiveHistory(History.load(configured.history_path))
            documents = DocumentService(configured, llama, live_corpus, rag)
            live_corpus.value = documents.prune_missing_uploads(live_corpus.value)
            await rag.rebuild(live_corpus.value)
            chat = ChatAgent(
                configured, llama, rag, live_corpus, live_history
            )
            runtime = ApplicationRuntime(
                configured,
                http,
                live_corpus,
                live_history,
                rag,
                documents,
                chat,
            )
            app.state.runtime = runtime
            yield
        finally:
            runtime = getattr(app.state, "runtime", None)
            if runtime is not None:
                await runtime.cancel_active()
            await http.aclose()

    app = FastAPI(
        title="Local RAG Chatbot",
        version="3.0.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(template_path)

    @app.post("/api/chat")
    async def chat(
        message: str = Form(...),
        file: UploadFile | None = File(None),
    ) -> StreamingResponse:
        runtime = _runtime(app)
        request_state = await runtime.claim_request()
        if request_state is None:
            raise HTTPException(status_code=409, detail="Another chat request is active")

        upload_name: str | None = None
        upload_content: bytes | None = None
        try:
            if file is not None and file.filename:
                upload_name = file.filename
                upload_content = await file.read(configured.max_upload_bytes + 1)
                await file.close()
        except BaseException:
            await runtime.release_request(request_state)
            raise

        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def pipeline() -> None:
            try:
                if request_state.cancel_event.is_set():
                    raise asyncio.CancelledError
                new_document_id: str | None = None
                if upload_name is not None and upload_content is not None:
                    queue.put_nowait({"status": "Đang xử lý tài liệu..."})
                    document = await runtime.documents.ingest(
                        upload_name,
                        upload_content,
                        request_state,
                    )
                    new_document_id = document.file_id
                    queue.put_nowait(
                        {
                            "status": (
                                f"Đã xử lý {document.file_name} "
                                f"({document.chunk_count} chunks)"
                            )
                        }
                    )
                queue.put_nowait({"status": "Đang tạo câu trả lời..."})
                async for content in runtime.chat.stream(
                    message, new_document_id=new_document_id
                ):
                    queue.put_nowait({"content": content})
                queue.put_nowait({"done": True})
            except asyncio.CancelledError:
                queue.put_nowait({"cancelled": True})
            except Exception as exc:
                detail = " ".join(str(exc).split())[:500] or exc.__class__.__name__
                queue.put_nowait({"error": detail})
            finally:
                request_state.task = None
                await runtime.release_request(request_state)

        async def events():
            producer = asyncio.create_task(
                pipeline(), name=f"chat-{request_state.request_id}"
            )
            request_state.task = producer
            try:
                yield _sse({"request_id": request_state.request_id})
                terminal = False
                while not terminal:
                    try:
                        event = await asyncio.wait_for(
                            queue.get(), timeout=heartbeat_interval
                        )
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    terminal = any(
                        event.get(key) is True
                        for key in ("done", "cancelled")
                    ) or "error" in event
                    yield _sse(event)
                await asyncio.gather(producer, return_exceptions=True)
            finally:
                async def cleanup() -> None:
                    request_state.cancel_event.set()
                    if not producer.done():
                        producer.cancel()
                    await asyncio.gather(producer, return_exceptions=True)
                    request_state.task = None
                    await runtime.release_request(request_state)

                cleanup_task = asyncio.create_task(cleanup())
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    await cleanup_task
                    raise

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/stop")
    async def stop() -> JSONResponse:
        cancelled = await _runtime(app).cancel_active()
        return JSONResponse({"status": "ok", "cancelled": cancelled})

    @app.get("/api/chat-history")
    async def history() -> JSONResponse:
        return JSONResponse(
            {"history": [item.to_dict() for item in _runtime(app).live_history.value.messages]}
        )

    @app.post("/api/clear-chat")
    async def clear_chat() -> JSONResponse:
        runtime = _runtime(app)
        await runtime.cancel_active()
        runtime.documents.clear()
        empty_history = History()
        empty_history.save(runtime.settings.history_path)
        runtime.live_history.value = empty_history
        return JSONResponse({"status": "ok"})

    @app.get("/api/documents")
    async def documents() -> JSONResponse:
        return JSONResponse(
            {
                "documents": [
                    {
                        "file_id": item.file_id,
                        "file_name": item.file_name,
                        "chunk_count": item.chunk_count,
                    }
                    for item in _runtime(app).live_corpus.value.documents
                ]
            }
        )

    @app.delete("/api/documents/{file_id}")
    async def delete_document(file_id: str) -> JSONResponse:
        runtime = _runtime(app)
        if await runtime.has_active_request():
            raise HTTPException(status_code=409, detail="A chat request is active")
        if not runtime.documents.delete(file_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return JSONResponse({"status": "ok"})

    @app.get("/api/documents/{file_id}/download")
    async def download_document(file_id: str) -> FileResponse:
        runtime = _runtime(app)
        document = next(
            (
                item
                for item in runtime.live_corpus.value.documents
                if item.file_id == file_id
            ),
            None,
        )
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        path = runtime.settings.uploads_dir / f"{document.file_id}_{document.file_name}"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(
            path,
            filename=document.file_name,
            media_type="application/octet-stream",
        )

    return app


def _runtime(app: FastAPI) -> ApplicationRuntime:
    runtime: ApplicationRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Application is not ready")
    return runtime


def _sse(event: dict[str, object]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
