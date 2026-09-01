"""Opt-in quality checks over the real Agents SDK document-agent path."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
import numpy as np
import pytest

from src.agent import AgentService
from src.config import Settings
from src.database import Database
from src.model_clients import LocalModelClients, build_agent_model
from src.models import DocumentRecord, StoredChunk
from src.rag import RagService, SnapshotStore
from src.sessions import SessionService


FIXTURE = Path(__file__).parent / "fixtures" / "agent_cases.json"
RUN_LIVE = os.getenv("RUN_LIVE_MODEL_TESTS") == "1"


def _cases() -> list[dict[str, object]]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_agent_fixture_is_broad_and_valid() -> None:
    cases = _cases()
    ids = [case["id"] for case in cases]
    categories = {case["category"] for case in cases}

    assert 40 <= len(cases) <= 60
    assert len(ids) == len(set(ids))
    assert {
        "direct",
        "upload_ack",
        "overview",
        "search",
        "comparison",
        "followup",
        "reference_error",
        "empty_retrieval",
        "safety",
    } <= categories
    for case in cases:
        assert isinstance(case["message"], str) and case["message"].strip()
        assert set(case["expected_choices"]) <= {"direct", "overview", "search"}
        assert case["expected_choices"]


def _records() -> tuple[list[DocumentRecord], list[StoredChunk]]:
    rows = [
        (
            "policy",
            "chinh-sach-cong-dong.pdf",
            "Chính sách coi phục vụ cộng đồng là trụ cột và nêu các nguyên tắc triển khai.",
            [("p. 1", "Phục vụ cộng đồng là một trong ba trụ cột của Nhà trường."), ("p. 2", "Nguyên tắc gồm phát triển bền vững và sử dụng nguồn lực hiệu quả.")],
        ),
        (
            "guide",
            "huong-dan-hoc-vu.docx",
            "Hướng dẫn quy trình đăng ký, học lại và cảnh báo học vụ.",
            [("p. 3", "Sinh viên đăng ký học lại trong thời hạn do phòng đào tạo công bố."), ("p. 5", "Cảnh báo học vụ được xử lý theo quy trình tư vấn và theo dõi kết quả.")],
        ),
        (
            "report",
            "bao-cao-2025.pdf",
            "Báo cáo tổng hợp kết quả, tăng trưởng và các số liệu năm 2025.",
            [("p. 4", "Báo cáo năm 2025 ghi nhận mức tăng trưởng 12 phần trăm."), ("p. 6", "Ba kết quả nổi bật gồm đào tạo, nghiên cứu và phục vụ cộng đồng.")],
        ),
        ("dup-a", "phu-luc.pdf", "Phụ lục A.", [("p. 1", "Nội dung phụ lục A.")]),
        ("dup-b", "phu-luc.pdf", "Phụ lục B.", [("p. 1", "Nội dung phụ lục B.")]),
    ]
    documents = [
        DocumentRecord(identifier, file_name, "application/octet-stream", "ready", overview, len(parts), "", 1.0, 1.0)
        for identifier, file_name, overview, parts in rows
    ]
    chunks = [
        StoredChunk(identifier, index, (reference,), text, None, None)
        for identifier, _file_name, _overview, parts in rows
        for index, (reference, text) in enumerate(parts)
    ]
    return documents, chunks


async def _runtime(
    settings: Settings, http: httpx.AsyncClient
) -> tuple[AgentService, SessionService, ThreadPoolExecutor]:
    database = Database(settings.database_path, settings.database_busy_timeout_ms)
    await database.initialize()
    sessions = SessionService(settings, database)
    models = LocalModelClients(
        settings,
        http,
        embedding_gate=asyncio.Semaphore(settings.embedding_concurrency),
        rerank_gate=asyncio.Semaphore(settings.rerank_concurrency),
    )
    executor = ThreadPoolExecutor(max_workers=settings.rag_cpu_workers)
    rag = RagService(
        models,
        cpu_executor=executor,
        embedding_batch_size=settings.embedding_batch_size,
        lexical_candidate_limit=settings.lexical_candidate_limit,
        semantic_candidate_limit=settings.semantic_candidate_limit,
        fused_candidate_limit=settings.fused_candidate_limit,
        final_chunk_limit=settings.final_chunk_limit,
    )
    documents, chunks = _records()
    vectors = np.asarray(await models.embed([chunk.text for chunk in chunks]), dtype=np.float32)
    snapshots = SnapshotStore(await rag.build(documents, chunks, vectors))
    return (
        AgentService(
            settings,
            snapshots,
            rag,
            sessions,
            responses_model=build_agent_model(settings, http),
        ),
        sessions,
        executor,
    )


def _input_item(role: str, text: str) -> dict[str, object]:
    if role == "user":
        return {"type": "message", "role": "user", "content": text}
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _mapping(item: object) -> Mapping[str, Any] | None:
    if isinstance(item, Mapping):
        return item
    model_dump = getattr(item, "model_dump", None)
    value = model_dump(exclude_unset=True) if callable(model_dump) else None
    return value if isinstance(value, Mapping) else None


def _tool_call(items: list[object]) -> tuple[str, dict[str, object]] | None:
    for item in reversed(items):
        value = _mapping(item)
        if value is None or value.get("type") != "function_call":
            continue
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            continue
        decoded = json.loads(arguments)
        if isinstance(decoded, dict):
            return name, decoded
    return None


@pytest.mark.live_model
@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_MODEL_TESTS=1")
@pytest.mark.asyncio
async def test_live_sdk_agent_decisions_and_grounded_final_answers() -> None:
    choice_total = choice_correct = 0
    follow_total = follow_correct = 0
    empty_claims = 0
    choice_misses: list[tuple[str, str]] = []

    with TemporaryDirectory() as directory:
        settings = Settings(data_dir=Path(directory) / "data")
        settings.ensure_dirs()
        timeout = httpx.Timeout(
            settings.http_read_timeout,
            connect=settings.http_connect_timeout,
            write=settings.http_write_timeout,
            pool=settings.http_pool_timeout,
        )
        async with httpx.AsyncClient(timeout=timeout) as http:
            service, sessions, executor = await _runtime(settings, http)
            try:
                for case in _cases():
                    session = await sessions.create()
                    history = case.get("history", [])
                    if isinstance(history, list):
                        await sessions.sdk_session(session.id).add_items(
                            [
                                _input_item(str(item["role"]), str(item["content"]))
                                for item in history
                                if isinstance(item, Mapping)
                            ]
                        )
                    events = [
                        event
                        async for event in service.stream(session.id, str(case["message"]))
                    ]
                    assert events and events[0].type == "start"
                    assert events[-1].type == "done", f"SDK run did not complete for {case['id']}"

                    items = await sessions.sdk_session(session.id).get_items(limit=-1)
                    call = _tool_call(items)
                    choice = "direct" if call is None else (
                        "overview" if call[0] == "get_document_overviews" else "search"
                    )
                    if case.get("score_choice", True):
                        choice_total += 1
                        choice_correct += choice in case["expected_choices"]
                        if choice not in case["expected_choices"]:
                            choice_misses.append((str(case["id"]), choice))

                    expected_ids = case.get("expected_file_ids")
                    if call is not None and expected_ids and case["category"] == "followup":
                        file_ids = call[1].get("file_ids", [])
                        follow_total += 1
                        follow_correct += isinstance(file_ids, list) and set(file_ids) == set(expected_ids)

                    messages = await sessions.messages(session.id)
                    final_text = messages[-1].content
                    assert final_text.strip()
                    if case.get("empty_result"):
                        lowered = final_text.casefold()
                        admits_absence = any(
                            phrase in lowered
                            for phrase in (
                                "không tìm thấy",
                                "không có thông tin",
                                "không được cung cấp",
                                "không thể xác nhận",
                            )
                        )
                        empty_claims += not admits_absence
            finally:
                await service.stop_all()
                executor.shutdown(wait=True)

    assert choice_total and choice_correct / choice_total >= 0.95
    assert follow_total and follow_correct / follow_total >= 0.90
    assert empty_claims == 0
    print(
        {
            "choice_accuracy": choice_correct / choice_total,
            "followup_file_accuracy": follow_correct / follow_total,
            "empty_result_claims": empty_claims,
            "choice_misses": choice_misses,
        }
    )


@pytest.mark.live_model
@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_MODEL_TESTS=1")
@pytest.mark.asyncio
async def test_live_reranker_score_direction() -> None:
    settings = Settings()
    cases = [
        (
            "Thủ đô của Việt Nam",
            "Hà Nội là thủ đô của Việt Nam.",
            ["Công thức làm bánh mì.", "Sao Mộc là một hành tinh."],
        ),
        (
            "Vietnamese student course registration",
            "Students register for courses through the academic portal.",
            ["A recipe for noodle soup.", "Weather on Mars."],
        ),
        (
            "chính sách phục vụ cộng đồng",
            "Nhà trường triển khai hoạt động gắn kết và phục vụ cộng đồng.",
            ["Hướng dẫn cài đặt phần mềm.", "Bảng giá linh kiện máy tính."],
        ),
    ]
    async with httpx.AsyncClient(timeout=60) as http:
        models = LocalModelClients(
            settings,
            http,
            embedding_gate=asyncio.Semaphore(1),
            rerank_gate=asyncio.Semaphore(1),
        )
        for query, positive, negatives in cases:
            scores = await models.rerank(query, [positive, *negatives])
            assert scores[0] > max(scores[1:])
