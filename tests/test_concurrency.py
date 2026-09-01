"""Concurrency and lifecycle contracts for the async HTTP runtime."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from src.agent import AgentEvent
from src.config import Settings
from src.main import create_app


@asynccontextmanager
async def _client(tmp_path: Path):
    app = create_app(
        Settings(data_dir=tmp_path / "data"),
        model_transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError(request.url))
        ),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield app, client


@pytest.mark.asyncio
async def test_four_sessions_run_without_global_conflict_and_duplicate_is_409(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        session_ids = [(await client.post("/api/sessions")).json()["id"] for _ in range(4)]
        entered = asyncio.Event()
        release = asyncio.Event()
        running = 0
        maximum = 0
        active: set[str] = set()

        async def stream(session_id: str, message: str):
            nonlocal running, maximum
            assert message == "hello"
            if session_id in active:
                raise ValueError("a chat run is already active for this session")
            active.add(session_id)
            running += 1
            maximum = max(maximum, running)
            if running == 4:
                entered.set()
            yield AgentEvent("start")
            await release.wait()
            running -= 1
            active.remove(session_id)
            yield AgentEvent("done")

        app.state.runtime.agent.stream = stream
        requests = [asyncio.create_task(client.post(f"/api/sessions/{sid}/chat", json={"message": "hello"})) for sid in session_ids]
        await asyncio.wait_for(entered.wait(), 1)
        duplicate = await client.post(f"/api/sessions/{session_ids[0]}/chat", json={"message": "hello"})
        release.set()
        responses = await asyncio.gather(*requests)
        assert [response.status_code for response in responses] == [200, 200, 200, 200]
        assert duplicate.status_code == 409
        assert maximum == 4


@pytest.mark.asyncio
async def test_session_delete_stops_the_exact_run_before_removing_metadata(tmp_path: Path) -> None:
    async with _client(tmp_path) as (app, client):
        session_id = (await client.post("/api/sessions")).json()["id"]
        stopped: list[str] = []

        async def stop_and_settle(value: str) -> None:
            stopped.append(value)

        app.state.runtime.agent.stop_and_settle = stop_and_settle
        response = await client.delete(f"/api/sessions/{session_id}")
        assert response.status_code == 204
        assert stopped == [session_id]
        assert (await client.get("/api/sessions")).json() == {"sessions": []}
