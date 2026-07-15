import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path

import httpx
import pytest

from src.config import Settings
from src.models import Chunk, Corpus, Document, History, Message
from src.main import create_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def _unexpected_model_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected model request: {request.url}")


@asynccontextmanager
async def _client(tmp_path: Path, *, heartbeat: float = 0.01):
    app = create_app(
        _settings(tmp_path),
        model_transport=httpx.MockTransport(_unexpected_model_request),
        heartbeat_interval=heartbeat,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield app, client


def _events(response: httpx.Response) -> list[dict[str, object]]:
    values = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            values.append(json.loads(line[6:]))
    return values


@pytest.mark.asyncio
async def test_empty_startup_and_shutdown_own_one_http_client(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path),
        model_transport=httpx.MockTransport(_unexpected_model_request),
    )
    runtime = None

    async with app.router.lifespan_context(app):
        runtime = app.state.runtime
        assert runtime.rag.chunk_count == 0
        assert runtime.http.is_closed is False

    assert runtime is not None and runtime.http.is_closed is True


@pytest.mark.asyncio
async def test_nonempty_startup_rebuilds_and_failure_aborts_readiness(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    document = Document("doc", "doc.pdf", "overview", 1)
    corpus = Corpus(
        [document], [Chunk("doc", "doc.pdf", 0, ["p. 1"], "text")]
    )
    corpus.save(settings.corpus_path)
    (settings.uploads_dir / "doc_doc.pdf").write_bytes(b"file")

    def embedding(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embedding"
        texts = json.loads(request.content)["content"]
        return httpx.Response(
            200,
            json=[
                {"index": index, "embedding": [[1.0, float(index + 1)]]}
                for index, _ in enumerate(texts)
            ],
        )

    app = create_app(settings, model_transport=httpx.MockTransport(embedding))
    async with app.router.lifespan_context(app):
        assert app.state.runtime.rag.chunk_count == 1

    failed = create_app(
        settings,
        model_transport=httpx.MockTransport(
            lambda request: httpx.Response(503, text="unavailable")
        ),
    )
    with pytest.raises(Exception, match="503"):
        async with failed.router.lifespan_context(failed):
            pass


@pytest.mark.asyncio
async def test_successful_sse_has_request_status_content_and_done(
    tmp_path: Path,
) -> None:
    async with _client(tmp_path) as (app, client):
        async def direct(message: str, *, new_document_id: str | None = None):
            assert message == "hello"
            assert new_document_id is None
            yield "Xin chào"

        app.state.runtime.chat.stream = direct
        response = await client.post("/api/chat", data={"message": "hello"})

        assert response.status_code == 200
        events = _events(response)
        assert "request_id" in events[0]
        assert events[1] == {"status": "Đang tạo câu trả lời..."}
        assert events[2] == {"content": "Xin chào"}
        assert events[3] == {"done": True}
        assert app.state.runtime.active is None
        assert response.headers["x-accel-buffering"] == "no"


@pytest.mark.asyncio
async def test_second_request_is_409_and_stop_clears_active_state(
    tmp_path: Path,
) -> None:
    async with _client(tmp_path) as (app, client):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def blocking(message: str, *, new_document_id: str | None = None):
            del message, new_document_id
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()
            yield "unreachable"

        app.state.runtime.chat.stream = blocking
        first = asyncio.create_task(client.post("/api/chat", data={"message": "one"}))
        await asyncio.wait_for(started.wait(), 2)

        second = await client.post("/api/chat", data={"message": "two"})
        assert second.status_code == 409
        assert (await client.delete("/api/documents/none")).status_code == 409

        stopped = await client.post("/api/stop")
        response = await asyncio.wait_for(first, 2)

        assert stopped.json() == {"status": "ok", "cancelled": True}
        assert {"cancelled": True} in _events(response)
        assert cleaned.is_set()
        assert app.state.runtime.active is None


@pytest.mark.asyncio
async def test_heartbeat_is_sse_comment(tmp_path: Path) -> None:
    async with _client(tmp_path, heartbeat=0.005) as (app, client):
        async def slow(message: str, *, new_document_id: str | None = None):
            del message, new_document_id
            await asyncio.sleep(0.03)
            yield "done"

        app.state.runtime.chat.stream = slow
        response = await client.post("/api/chat", data={"message": "wait"})

        assert ": heartbeat\n\n" in response.text
        assert _events(response)[-1] == {"done": True}


@pytest.mark.asyncio
async def test_upload_status_order_and_committed_document(
    tmp_path: Path,
) -> None:
    async with _client(tmp_path) as (app, client):
        runtime = app.state.runtime

        async def ingest(name, content, state):
            del content, state
            document = Document("new", Path(name).name, "overview", 1)
            chunk = Chunk("new", document.file_name, 0, ["p. 1"], "text")
            runtime.live_corpus.value = runtime.live_corpus.value.with_document(
                document, [chunk]
            )
            return document

        async def answer(message: str, *, new_document_id: str | None = None):
            assert new_document_id == "new"
            yield f"ack:{message}"

        runtime.documents.ingest = ingest
        runtime.chat.stream = answer
        response = await client.post(
            "/api/chat",
            data={"message": "read it"},
            files={"file": ("report.pdf", b"file", "application/pdf")},
        )

        events = _events(response)
        assert [next(iter(event)) for event in events] == [
            "request_id",
            "status",
            "status",
            "status",
            "content",
            "done",
        ]
        assert "Đang xử lý" in events[1]["status"]
        assert events[2]["status"] == "Đã xử lý report.pdf (1 đoạn)"
        assert (await client.get("/api/documents")).json()["documents"][0]["file_id"] == "new"


@pytest.mark.asyncio
async def test_stop_after_ingest_commit_preserves_document(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        runtime = app.state.runtime
        committed = asyncio.Event()
        chat_started = asyncio.Event()

        async def ingest(name, content, state):
            del content, state
            document = Document("committed", Path(name).name, "overview", 1)
            chunk = Chunk("committed", document.file_name, 0, ["p. 1"], "text")
            runtime.live_corpus.value = runtime.live_corpus.value.with_document(
                document, [chunk]
            )
            committed.set()
            return document

        async def blocking(message: str, *, new_document_id: str | None = None):
            del message, new_document_id
            chat_started.set()
            await asyncio.Future()
            yield "unreachable"

        runtime.documents.ingest = ingest
        runtime.chat.stream = blocking
        request = asyncio.create_task(
            client.post(
                "/api/chat",
                data={"message": "question"},
                files={"file": ("report.pdf", b"file", "application/pdf")},
            )
        )
        await asyncio.wait_for(committed.wait(), 2)
        await asyncio.wait_for(chat_started.wait(), 2)
        await client.post("/api/stop")
        await request

        documents = (await client.get("/api/documents")).json()["documents"]
        assert [item["file_id"] for item in documents] == ["committed"]


@pytest.mark.asyncio
async def test_failed_ingest_never_appears_in_documents(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        async def fail(name, content, state):
            del name, content, state
            raise RuntimeError("parse failed")

        app.state.runtime.documents.ingest = fail
        response = await client.post(
            "/api/chat",
            data={"message": "read"},
            files={"file": ("bad.pdf", b"file", "application/pdf")},
        )

        assert "parse failed" in _events(response)[-1]["error"]
        assert (await client.get("/api/documents")).json() == {"documents": []}
        assert app.state.runtime.active is None


@pytest.mark.asyncio
async def test_disconnect_cancels_pipeline_and_releases_slot(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def blocking(message: str, *, new_document_id: str | None = None):
            del message, new_document_id
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()
            yield "unreachable"

        app.state.runtime.chat.stream = blocking
        request = asyncio.create_task(client.post("/api/chat", data={"message": "one"}))
        await asyncio.wait_for(started.wait(), 2)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.wait_for(cleaned.wait(), 2)
        for _ in range(100):
            if app.state.runtime.active is None:
                break
            await asyncio.sleep(0.01)
        assert app.state.runtime.active is None


@pytest.mark.asyncio
async def test_history_documents_download_delete_and_clear_contracts(
    tmp_path: Path,
) -> None:
    async with _client(tmp_path) as (app, client):
        runtime = app.state.runtime
        document = Document("doc", "report.pdf", "overview", 1)
        chunk = Chunk("doc", "report.pdf", 0, ["p. 1"], "text")
        runtime.live_corpus.value = Corpus([document], [chunk])
        runtime.live_corpus.value.save(runtime.settings.corpus_path)
        upload = runtime.settings.uploads_dir / "doc_report.pdf"
        upload.write_bytes(b"download-content")
        runtime.live_history.value = History(
            [Message("user", "hi"), Message("assistant", "hello")]
        )
        runtime.live_history.value.save(runtime.settings.history_path)

        assert (await client.get("/api/chat-history")).json() == {
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        assert (await client.get("/api/documents")).json()["documents"] == [
            {"file_id": "doc", "file_name": "report.pdf", "chunk_count": 1}
        ]
        downloaded = await client.get("/api/documents/doc/download")
        assert downloaded.content == b"download-content"
        assert "report.pdf" in downloaded.headers["content-disposition"]

        assert (await client.delete("/api/documents/doc")).status_code == 200
        assert not upload.exists()
        assert runtime.live_corpus.value == Corpus()

        runtime.live_history.value = History([Message("user", "again")])
        runtime.live_history.value.save(runtime.settings.history_path)
        assert (await client.post("/api/clear-chat")).json() == {"status": "ok"}
        assert runtime.live_history.value == History()
        assert History.load(runtime.settings.history_path) == History()


def test_fastapi_has_no_parser_cuda_or_model_cleanup_paths() -> None:
    source = Path("src/main.py").read_text(encoding="utf-8").lower()

    for forbidden in (
        "liteparse",
        "tokenizers",
        "semantic_text_splitter",
        "tesseract",
        "cuda",
        "empty_cache",
        "kill model",
        ".release()",
    ):
        assert forbidden not in source
