import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import pytest

from src.chat import ChatAgent, FINAL_INSTRUCTION, LiveHistory, TOOLS
from src.config import Settings
from src.documents import LiveCorpus
from src.llama import ContentEvent, LlamaClient, ToolCallEvent
from src.models import Chunk, Corpus, Document, History, Message


FIXTURE = Path(__file__).parent / "fixtures" / "agent_cases.json"
RUN_LIVE = os.getenv("RUN_LIVE_MODEL_TEST") == "1"


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


class _EvalRag:
    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    async def search(
        self, queries: list[str], file_ids: list[str], limit: int
    ) -> list[Chunk]:
        joined = " ".join(queries).casefold()
        if "sao hỏa" in joined or "sao hoa" in joined:
            return []
        allowed = set(file_ids)
        return [
            chunk
            for chunk in self._corpus.chunks
            if chunk.file_id in allowed
        ][:limit]


def _eval_corpus() -> Corpus:
    documents = [
        Document(
            "policy",
            "chinh-sach-cong-dong.pdf",
            "Chính sách coi phục vụ cộng đồng là trụ cột và nêu các nguyên tắc triển khai.",
            2,
        ),
        Document(
            "guide",
            "huong-dan-hoc-vu.docx",
            "Hướng dẫn quy trình đăng ký, học lại và cảnh báo học vụ.",
            2,
        ),
        Document(
            "report",
            "bao-cao-2025.pdf",
            "Báo cáo tổng hợp kết quả, tăng trưởng và các số liệu năm 2025.",
            2,
        ),
        Document("dup-a", "phu-luc.pdf", "Phụ lục A.", 1),
        Document("dup-b", "phu-luc.pdf", "Phụ lục B.", 1),
    ]
    chunks = [
        Chunk("policy", documents[0].file_name, 0, ["p. 1"], "Phục vụ cộng đồng là một trong ba trụ cột của Nhà trường."),
        Chunk("policy", documents[0].file_name, 1, ["p. 2"], "Nguyên tắc gồm phát triển bền vững và sử dụng nguồn lực hiệu quả."),
        Chunk("guide", documents[1].file_name, 0, ["p. 3"], "Sinh viên đăng ký học lại trong thời hạn do phòng đào tạo công bố."),
        Chunk("guide", documents[1].file_name, 1, ["p. 5"], "Cảnh báo học vụ được xử lý theo quy trình tư vấn và theo dõi kết quả."),
        Chunk("report", documents[2].file_name, 0, ["p. 4"], "Báo cáo năm 2025 ghi nhận mức tăng trưởng 12 phần trăm."),
        Chunk("report", documents[2].file_name, 1, ["p. 6"], "Ba kết quả nổi bật gồm đào tạo, nghiên cứu và phục vụ cộng đồng."),
        Chunk("dup-a", documents[3].file_name, 0, ["p. 1"], "Nội dung phụ lục A."),
        Chunk("dup-b", documents[4].file_name, 0, ["p. 1"], "Nội dung phụ lục B."),
    ]
    return Corpus(documents, chunks)


def _choice(events: list[object]) -> tuple[str, ToolCallEvent | None]:
    content = [event for event in events if isinstance(event, ContentEvent)]
    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    if content and calls:
        raise AssertionError("mixed content/tool protocol")
    if content:
        assert "".join(event.content for event in content).strip()
        return "direct", None
    assert len(calls) == 1
    name = calls[0].name
    assert name in {"get_document_overviews", "search_documents"}
    return ("overview" if name == "get_document_overviews" else "search"), calls[0]


@pytest.mark.live_model
@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_MODEL_TEST=1")
@pytest.mark.asyncio
async def test_live_agent_decisions_and_final_protocol() -> None:
    corpus = _eval_corpus()
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
            llama = LlamaClient(
                http, settings.llm_url, settings.embed_url, settings.rerank_url
            )
            for case in _cases():
                history = History(
                    [Message.from_dict(item) for item in case.get("history", [])]
                )
                agent = ChatAgent(
                    settings,
                    llama,
                    _EvalRag(corpus),
                    LiveCorpus(corpus),
                    LiveHistory(history),
                )
                messages = agent._first_messages(
                    case["message"], case.get("new_document_id")
                )
                events = [
                    event
                    async for event in llama.stream_chat(
                        messages, tools=TOOLS, tool_choice="auto"
                    )
                ]
                choice, call = _choice(events)
                if case.get("score_choice", True):
                    choice_total += 1
                    choice_correct += choice in case["expected_choices"]
                    if choice not in case["expected_choices"]:
                        choice_misses.append((case["id"], choice))

                if call is None:
                    continue
                tool_result = await agent._execute_tool(call)
                expected_ids = case.get("expected_file_ids")
                if expected_ids:
                    arguments = json.loads(call.arguments)
                    selected, error = agent._resolve_file_ids(arguments["file_ids"])
                    if case["category"] == "followup":
                        follow_total += 1
                        follow_correct += error is None and set(selected) == set(expected_ids)

                final_messages = messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": call.arguments,
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": tool_result,
                    },
                    {"role": "user", "content": FINAL_INSTRUCTION},
                ]
                final_events = [
                    event async for event in llama.stream_chat(final_messages)
                ]
                assert final_events, f"empty final stream for {case['id']}"
                assert all(
                    isinstance(event, ContentEvent) for event in final_events
                ), f"invalid final protocol for {case['id']}: {final_events!r}"
                final_text = "".join(event.content for event in final_events).strip()
                assert final_text
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
@pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_MODEL_TEST=1")
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
        llama = LlamaClient(
            http, settings.llm_url, settings.embed_url, settings.rerank_url
        )
        for query, positive, negatives in cases:
            scores = await llama.rerank(query, [positive, *negatives])
            assert scores[0] > max(scores[1:])
