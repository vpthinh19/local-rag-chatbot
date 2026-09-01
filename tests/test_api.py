"""HTTP contracts for the session/document API cutover."""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from src.agent import AgentEvent
from src.config import Settings
from src.main import create_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@asynccontextmanager
async def _client(tmp_path: Path, *, heartbeat: float = 0.005):
    app = create_app(
        _settings(tmp_path),
        model_transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError(request.url))
        ),
        heartbeat_interval=heartbeat,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield app, client


def _events(response: httpx.Response) -> list[dict[str, object]]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.mark.asyncio
async def test_session_crud_and_complete_message_projection(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        created = await client.post("/api/sessions")
        assert created.status_code == 201
        session = created.json()
        session_id = session["id"]
        assert session["title"] == "New chat"
        renamed = await client.patch(f"/api/sessions/{session_id}", json={"title": "Project notes"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Project notes"
        await app.state.runtime.sessions.sdk_session(session_id).add_items([
            {"role": "user", "content": "question"},
            {"type": "function_call", "name": "hidden"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "incomplete"},
        ])
        assert (await client.get(f"/api/sessions/{session_id}/messages")).json() == {
            "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
        }
        assert (await client.get("/api/sessions")).json()["sessions"][0]["id"] == session_id
        assert (await client.delete(f"/api/sessions/{session_id}")).status_code == 204
        assert (await client.get(f"/api/sessions/{session_id}/messages")).status_code == 404


@pytest.mark.asyncio
async def test_chat_sse_uses_named_application_events_and_heartbeat(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        session_id = (await client.post("/api/sessions")).json()["id"]

        async def stream(session: str, message: str):
            assert (session, message) == (session_id, "hello")
            yield AgentEvent("start")
            await asyncio.sleep(0.015)
            yield AgentEvent("status", "looking up")
            yield AgentEvent("delta", "answer")
            yield AgentEvent("done")

        app.state.runtime.agent.stream = stream
        response = await client.post(f"/api/sessions/{session_id}/chat", json={"message": "hello"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert _events(response) == [
            {"type": "start"}, {"type": "status", "text": "looking up"},
            {"type": "delta", "text": "answer"}, {"type": "done"},
        ]
        assert ": heartbeat\n\n" in response.text


@pytest.mark.asyncio
async def test_chat_validation_is_json_4xx_before_stream_and_stream_errors_are_events(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        session_id = (await client.post("/api/sessions")).json()["id"]
        assert (await client.post(f"/api/sessions/{session_id}/chat", json={"message": " "})).status_code == 400
        assert (await client.post("/api/sessions/missing/chat", json={"message": "hello"})).status_code == 404

        async def fail(session: str, message: str):
            del session, message
            yield AgentEvent("start")
            yield AgentEvent("error")

        app.state.runtime.agent.stream = fail
        response = await client.post(f"/api/sessions/{session_id}/chat", json={"message": "hello"})
        assert _events(response) == [{"type": "start"}, {"type": "error"}]


@pytest.mark.asyncio
async def test_document_upload_list_download_retry_and_durable_delete(tmp_path: Path) -> None:
    async with _client(tmp_path) as (_app, client):
        uploaded = await client.post("/api/documents", files={"file": ("report.pdf", b"pdf", "application/pdf")})
        assert uploaded.status_code == 202
        document = uploaded.json()
        assert document["status"] == "processing"
        document_id = document["id"]
        assert (await client.get("/api/documents")).json()["documents"][0]["id"] == document_id
        download = await client.get(f"/api/documents/{document_id}/download")
        assert download.status_code == 200 and download.content == b"pdf"
        assert "report.pdf" in download.headers["content-disposition"]
        assert (await client.post(f"/api/documents/{document_id}/retry")).status_code == 409
        deleted = await client.delete(f"/api/documents/{document_id}")
        assert deleted.status_code == 202 and deleted.json()["status"] == "deleting"
        assert (await client.get(f"/api/documents/{document_id}/download")).status_code == 409


@pytest.mark.asyncio
async def test_document_upload_rejects_non_multipart_and_invalid_upload(tmp_path: Path) -> None:
    async with _client(tmp_path) as (_app, client):
        assert (await client.post("/api/documents", json={"file": "nope"})).status_code == 422
        assert (await client.post("/api/documents", files={"file": ("bad.txt", b"x", "text/plain")})).status_code == 400
