import asyncio
import json
from pathlib import Path

import pytest

from src.chat import ChatAgent, ChatProtocolError, FINAL_INSTRUCTION, LiveHistory
from src.config import Settings
from src.documents import LiveCorpus
from src.llama import ContentEvent, ToolCallEvent
from src.models import Chunk, Corpus, Document, History, Message


class FakeLlama:
    def __init__(self, responses: list[list[object] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def stream_chat(self, messages, tools=None, tool_choice=None):
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        for event in response:
            await asyncio.sleep(0)
            yield event


class FakeRag:
    def __init__(self, results: list[Chunk] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[str], list[str], int]] = []

    async def search(
        self, queries: list[str], file_ids: list[str], limit: int
    ) -> list[Chunk]:
        self.calls.append((queries, file_ids, limit))
        return list(self.results)


def _corpus(duplicate_name: bool = False) -> Corpus:
    second_name = "report.pdf" if duplicate_name else "guide.pdf"
    return Corpus(
        [
            Document("doc-a", "report.pdf", "Overview A", 1),
            Document("doc-b", second_name, "Overview B", 1),
        ],
        [
            Chunk("doc-a", "report.pdf", 0, ["p. 1"], "Alpha fact"),
            Chunk("doc-b", second_name, 0, ["p. 2"], "Beta fact"),
        ],
    )


def _agent(
    tmp_path: Path,
    llama: FakeLlama,
    *,
    corpus: Corpus | None = None,
    history: History | None = None,
    rag: FakeRag | None = None,
) -> tuple[ChatAgent, LiveHistory, FakeRag, Settings]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    live_history = LiveHistory(history or History())
    fake_rag = rag or FakeRag()
    agent = ChatAgent(
        settings,
        llama,
        fake_rag,
        LiveCorpus(corpus or _corpus()),
        live_history,
    )
    return agent, live_history, fake_rag, settings


async def _collect(agent: ChatAgent, message: str, new_id: str | None = None) -> str:
    return "".join(
        [part async for part in agent.stream(message, new_document_id=new_id)]
    )


@pytest.mark.asyncio
async def test_greeting_is_one_direct_model_call(tmp_path: Path) -> None:
    llama = FakeLlama([[ContentEvent("Xin "), ContentEvent("chào!")]])
    agent, history, rag, settings = _agent(tmp_path, llama)

    answer = await _collect(agent, "Xin chào")

    assert answer == "Xin chào!"
    assert len(llama.calls) == 1
    assert llama.calls[0]["tool_choice"] == "auto"
    assert rag.calls == []
    assert history.value.messages == [
        Message("user", "Xin chào"),
        Message("assistant", "Xin chào!"),
    ]
    assert History.load(settings.history_path) == history.value


@pytest.mark.asyncio
async def test_empty_or_oversized_message_never_reaches_model(tmp_path: Path) -> None:
    llama = FakeLlama([])
    agent, history, rag, settings = _agent(tmp_path, llama)

    with pytest.raises(ValueError, match="empty"):
        await _collect(agent, "  ")
    with pytest.raises(ValueError, match="size"):
        await _collect(agent, "x" * (settings.max_message_chars + 1))

    assert llama.calls == []
    assert rag.calls == []
    assert history.value == History()


def test_agent_prompt_handles_typo_and_requires_exact_evidence(tmp_path: Path) -> None:
    agent, _, _, _ = _agent(tmp_path, FakeLlama([]))

    system = agent._first_messages("han nop ngafy nao", None)[0]["content"]

    assert "không dấu hoặc sai chính tả nhẹ" in system
    assert "phải dùng search" in system
    assert "chép chính xác" in FINAL_INSTRUCTION


@pytest.mark.asyncio
async def test_upload_acknowledgement_stays_direct(tmp_path: Path) -> None:
    llama = FakeLlama([[ContentEvent("Đã đọc và sẵn sàng.")]])
    agent, _, rag, _ = _agent(tmp_path, llama)

    await _collect(agent, "Hãy đọc file này", "doc-a")

    system = llama.calls[0]["messages"][0]["content"]
    assert "doc-a" in system
    assert "lượt đầu chỉ được chứa tool call" in system
    assert rag.calls == []


@pytest.mark.asyncio
async def test_overview_tool_then_final_answer_keeps_history_clean(tmp_path: Path) -> None:
    call = ToolCallEvent(0, "call-1", "get_document_overviews", '{"file_ids":["doc-a"]}')
    llama = FakeLlama([[call], [ContentEvent("Tóm tắt cuối")]])
    agent, history, _, _ = _agent(tmp_path, llama)

    answer = await _collect(agent, "Tóm tắt report.pdf")

    assert answer == "Tóm tắt cuối"
    assert len(llama.calls) == 2
    final_messages = llama.calls[1]["messages"]
    tool_message = next(item for item in final_messages if item["role"] == "tool")
    assert "Overview A" in tool_message["content"]
    assert history.value.messages == [
        Message("user", "Tóm tắt report.pdf"),
        Message("assistant", "Tóm tắt cuối"),
    ]
    assert "Overview A" not in json.dumps(history.value.to_dict(), ensure_ascii=False)


@pytest.mark.asyncio
async def test_search_tool_uses_rewritten_queries_and_supplies_citations(
    tmp_path: Path,
) -> None:
    call = ToolCallEvent(
        0,
        "call-2",
        "search_documents",
        '{"queries":["alpha standalone","alpha detail"],"file_ids":["doc-a"],"limit":2}',
    )
    chunk = Chunk("doc-a", "report.pdf", 0, ["p. 1"], "Alpha fact")
    rag = FakeRag([chunk])
    llama = FakeLlama([[call], [ContentEvent("Alpha [report.pdf, p. 1]")]])
    agent, history, _, _ = _agent(tmp_path, llama, rag=rag)

    answer = await _collect(agent, "Chi tiết alpha?")

    assert rag.calls == [(["alpha standalone", "alpha detail"], ["doc-a"], 2)]
    assert answer == "Alpha [report.pdf, p. 1]"
    final_tool = next(
        item for item in llama.calls[1]["messages"] if item["role"] == "tool"
    )
    assert "Alpha fact" in final_tool["content"]
    assert "citation" in final_tool["content"]
    assert "Alpha fact" not in json.dumps(history.value.to_dict())


@pytest.mark.asyncio
async def test_recent_clean_history_and_catalog_are_in_first_prompt(tmp_path: Path) -> None:
    previous = History([Message("user", "Câu trước"), Message("assistant", "Trả lời trước")])
    llama = FakeLlama([[ContentEvent("Tiếp tục")]])
    agent, _, _, _ = _agent(tmp_path, llama, history=previous)

    await _collect(agent, "Còn gì nữa?")

    messages = llama.calls[0]["messages"]
    assert {item["content"] for item in messages[1:-1]} == {
        "Câu trước",
        "Trả lời trước",
    }
    assert "report.pdf" in messages[0]["content"]
    assert "guide.pdf" in messages[0]["content"]


@pytest.mark.asyncio
async def test_unique_filename_is_normalized_to_file_id(tmp_path: Path) -> None:
    call = ToolCallEvent(
        0,
        "call-name",
        "get_document_overviews",
        '{"file_ids":["REPORT.PDF"]}',
    )
    llama = FakeLlama([[call], [ContentEvent("Đã tìm thấy")]])
    agent, _, _, _ = _agent(tmp_path, llama)

    await _collect(agent, "Tóm tắt report")

    tool = next(item for item in llama.calls[1]["messages"] if item["role"] == "tool")
    assert '"file_id": "doc-a"' in tool["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["missing.pdf", "report.pdf"])
async def test_unknown_or_ambiguous_reference_gets_structured_clarification(
    tmp_path: Path, reference: str
) -> None:
    call = ToolCallEvent(
        0,
        "call-ref",
        "get_document_overviews",
        json.dumps({"file_ids": [reference]}),
    )
    llama = FakeLlama([[call], [ContentEvent("Bạn vui lòng chọn lại file.")]])
    agent, history, _, _ = _agent(
        tmp_path, llama, corpus=_corpus(duplicate_name=reference == "report.pdf")
    )

    await _collect(agent, "Tóm tắt file")

    tool = next(item for item in llama.calls[1]["messages"] if item["role"] == "tool")
    assert '"status": "error"' in tool["content"]
    expected = "ambiguous_file" if reference == "report.pdf" else "unknown_file"
    assert expected in tool["content"]
    assert len(history.value.messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name, arguments",
    [
        ("unknown_tool", "{}"),
        ("search_documents", "not-json"),
        ("search_documents", '{"queries":[],"file_ids":["doc-a"],"limit":2}'),
        ("search_documents", '{"queries":["q"],"file_ids":["doc-a"],"limit":7}'),
        ("get_document_overviews", '{"file_ids":[]}'),
    ],
)
async def test_invalid_tool_protocol_is_rejected_without_history(
    tmp_path: Path, name: str, arguments: str
) -> None:
    llama = FakeLlama([[ToolCallEvent(0, "bad", name, arguments)]])
    agent, history, rag, _ = _agent(tmp_path, llama)

    with pytest.raises(ChatProtocolError):
        await _collect(agent, "question")

    assert history.value == History()
    assert rag.calls == []


@pytest.mark.asyncio
async def test_empty_retrieval_supplies_no_results_policy(tmp_path: Path) -> None:
    call = ToolCallEvent(
        0,
        "empty",
        "search_documents",
        '{"queries":["missing fact"],"file_ids":["doc-a"],"limit":2}',
    )
    llama = FakeLlama([[call], [ContentEvent("Không tìm thấy thông tin.")]])
    agent, _, _, _ = _agent(tmp_path, llama, rag=FakeRag([]))

    await _collect(agent, "Missing?")

    tool = next(item for item in llama.calls[1]["messages"] if item["role"] == "tool")
    assert "no_usable_results" in tool["content"]
    assert "must_not_claim_document_facts" in tool["content"]


@pytest.mark.asyncio
async def test_mixed_or_multiple_tool_calls_are_rejected(tmp_path: Path) -> None:
    mixed = FakeLlama(
        [[ContentEvent("partial"), ToolCallEvent(0, "one", "get_document_overviews", '{"file_ids":["doc-a"]}')]]
    )
    agent, history, _, _ = _agent(tmp_path, mixed)
    with pytest.raises(ChatProtocolError, match="mixed"):
        await _collect(agent, "question")
    assert history.value == History()

    multiple = FakeLlama(
        [[
            ToolCallEvent(0, "one", "get_document_overviews", '{"file_ids":["doc-a"]}'),
            ToolCallEvent(1, "two", "get_document_overviews", '{"file_ids":["doc-b"]}'),
        ]]
    )
    agent, history, _, _ = _agent(tmp_path / "other", multiple)
    with pytest.raises(ChatProtocolError, match="one tool"):
        await _collect(agent, "question")
    assert history.value == History()


@pytest.mark.asyncio
async def test_second_tool_call_is_rejected(tmp_path: Path) -> None:
    first = ToolCallEvent(0, "one", "get_document_overviews", '{"file_ids":["doc-a"]}')
    second = ToolCallEvent(0, "two", "search_documents", '{"queries":["x"],"file_ids":["doc-a"],"limit":1}')
    llama = FakeLlama([[first], [second]])
    agent, history, _, _ = _agent(tmp_path, llama)

    with pytest.raises(ChatProtocolError, match="second tool"):
        await _collect(agent, "question")

    assert history.value == History()


@pytest.mark.asyncio
async def test_cancelled_partial_or_failed_stream_leaves_history_unchanged(
    tmp_path: Path,
) -> None:
    llama = FakeLlama([[ContentEvent("partial"), ContentEvent("more")]])
    agent, history, _, _ = _agent(tmp_path, llama)
    stream = agent.stream("question")
    assert await anext(stream) == "partial"
    await stream.aclose()
    assert history.value == History()

    failed = FakeLlama([RuntimeError("model failed")])
    failed_agent, failed_history, _, _ = _agent(tmp_path / "failed", failed)
    with pytest.raises(RuntimeError, match="failed"):
        await _collect(failed_agent, "question")
    assert failed_history.value == History()


@pytest.mark.asyncio
async def test_chat_cancel_does_not_remove_committed_document(tmp_path: Path) -> None:
    corpus = _corpus()
    llama = FakeLlama([[ContentEvent("partial"), ContentEvent("more")]])
    agent, history, _, _ = _agent(tmp_path, llama, corpus=corpus)
    stream = agent.stream("question", new_document_id="doc-a")
    assert await anext(stream) == "partial"

    await stream.aclose()

    assert agent.corpus == corpus
    assert history.value == History()
