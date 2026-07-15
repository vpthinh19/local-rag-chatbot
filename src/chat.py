"""A bounded one-tool document agent centered on the LLM."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Any

from src.config import Settings
from src.documents import LiveCorpus
from src.llama import ContentEvent, LlamaClient, ToolCallEvent
from src.models import Corpus, History, Message
from src.rag import RagIndex


class ChatProtocolError(RuntimeError):
    """The model produced an invalid or unsafe agent protocol response."""


@dataclass(slots=True)
class LiveHistory:
    """Mutable holder for the currently committed history snapshot."""

    value: History


TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_document_overviews",
            "description": (
                "Lấy overview đã lưu để tóm tắt, lập dàn ý hoặc so sánh "
            "khái quát tài liệu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    }
                },
                "required": ["file_ids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Tìm đoạn trích cho câu hỏi chi tiết hoặc dữ kiện cụ thể. "
                "Viết lại tối đa ba truy vấn độc lập khi cần."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "file_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 8,
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 6},
                },
                "required": ["queries", "file_ids", "limit"],
                "additionalProperties": False,
            },
        },
    },
]

FINAL_INSTRUCTION = (
    "Hãy trả lời yêu cầu ban đầu ngay bây giờ. Chỉ dùng dữ liệu trong tool result; "
    "chép chính xác tên riêng, con số, ngày tháng và địa điểm, không thay thế bằng "
    "kiến thức hay phỏng đoán. "
    "nếu là kết quả search thì thêm citation tên file và refs, nếu status là error "
    "thì hỏi người dùng làm rõ, nếu không có kết quả thì nói không tìm thấy và "
    "không suy đoán. Không gọi thêm công cụ."
)


class ChatAgent:
    """Run a bounded direct-answer or single-tool LLM interaction."""

    def __init__(
        self,
        settings: Settings,
        llama: LlamaClient,
        rag: RagIndex,
        live_corpus: LiveCorpus,
        live_history: LiveHistory,
    ) -> None:
        """Bind model, retrieval, corpus, and history state."""
        self._settings = settings
        self._llama = llama
        self._rag = rag
        self._corpus = live_corpus
        self._history = live_history

    @property
    def corpus(self) -> Corpus:
        """Expose the current corpus snapshot for read-only callers."""
        return self._corpus.value

    async def stream(
        self, message: str, *, new_document_id: str | None = None
    ) -> AsyncIterator[str]:
        """Stream one answer and persist it only after successful completion."""
        user_message = message.strip()
        if not user_message:
            raise ValueError("chat message must not be empty")
        if len(user_message) > self._settings.max_message_chars:
            raise ValueError("chat message exceeds the size limit")
        messages = self._first_messages(user_message, new_document_id)
        content_parts: list[str] = []
        tool_calls: list[ToolCallEvent] = []

        async for event in self._llama.stream_chat(
            messages, tools=TOOLS, tool_choice="auto"
        ):
            if isinstance(event, ContentEvent):
                if tool_calls:
                    raise ChatProtocolError("mixed content and tool-call response")
                content_parts.append(event.content)
                yield event.content
            elif isinstance(event, ToolCallEvent):
                if content_parts:
                    raise ChatProtocolError("mixed content and tool-call response")
                tool_calls.append(event)
            else:
                raise ChatProtocolError("unknown chat stream event")

        # A first-turn answer and a tool call are mutually exclusive protocols.
        if content_parts:
            answer = "".join(content_parts)
            self._persist_turn(user_message, answer)
            return
        if len(tool_calls) != 1:
            if tool_calls:
                raise ChatProtocolError("the agent may call only one tool")
            raise ChatProtocolError("model returned neither content nor a tool call")

        call = tool_calls[0]
        tool_result = await self._execute_tool(call)
        # Tool context is request-local; only the final plain-text turn is persisted.
        local_messages = list(messages)
        local_messages.extend(
            [
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
        )

        final_parts: list[str] = []
        async for event in self._llama.stream_chat(local_messages):
            if isinstance(event, ToolCallEvent):
                raise ChatProtocolError("second tool call is not allowed")
            if not isinstance(event, ContentEvent):
                raise ChatProtocolError("unknown final stream event")
            final_parts.append(event.content)
            yield event.content
        final_answer = "".join(final_parts)
        if not final_answer.strip():
            raise ChatProtocolError("model returned an empty final answer")
        self._persist_turn(user_message, final_answer)

    def _first_messages(
        self, user_message: str, new_document_id: str | None
    ) -> list[dict[str, object]]:
        """Build the first LLM turn with catalog, context, and protocol rules."""
        catalog = [
            {
                "file_id": document.file_id,
                "file_name": document.file_name,
                "chunk_count": document.chunk_count,
            }
            for document in self._corpus.value.documents
        ]
        new_document = (
            new_document_id
            if new_document_id in {item.file_id for item in self._corpus.value.documents}
            else None
        )
        system = (
            "Bạn là trợ lý tài liệu thân thiện. Trả lời trực tiếp lời chào, hội thoại "
            "thông thường và yêu cầu chỉ đọc/giữ file. Khi cần nội dung tài liệu, gọi "
            "đúng một công cụ: overview cho tóm tắt/dàn ý, search cho dữ kiện chi tiết. "
            "Người dùng có thể viết tiếng Việt không dấu hoặc sai chính tả nhẹ; hãy suy "
            "ra ý định. Khi họ hỏi dữ kiện và có tài liệu phù hợp, phải dùng search thay "
            "vì hỏi lại có muốn tìm kiếm hay không. "
            "QUY TẮC PROTOCOL: nếu gọi công cụ, lượt đầu chỉ được chứa tool call; tuyệt "
            "đối không viết câu dẫn, giải thích hay content trước/sau tool call. "
            "Không tự bịa nội dung tài liệu. Sau search, trích dẫn ngắn bằng tên file "
            "và refs được cung cấp. Không gọi công cụ lần hai.\n"
            f"Tài liệu sẵn sàng: {json.dumps(catalog, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Tài liệu vừa được ingest: {new_document or 'none'}"
        )
        history = self._recent_history()
        return [
            {"role": "system", "content": system},
            *[message.to_dict() for message in history],
            {"role": "user", "content": user_message},
        ]

    def _recent_history(self) -> list[Message]:
        """Select a bounded suffix of recent persisted conversation."""
        selected: list[Message] = []
        size = 0
        for message in reversed(self._history.value.messages[-12:]):
            content = message.content[-6_000:]
            if selected and size + len(content) > 12_000:
                break
            selected.append(Message(message.role, content))
            size += len(content)
        return list(reversed(selected))

    async def _execute_tool(self, call: ToolCallEvent) -> str:
        """Validate and execute the one tool selected by the LLM."""
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ChatProtocolError("tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ChatProtocolError("tool arguments must be an object")

        if call.name == "get_document_overviews":
            self._require_exact_keys(arguments, {"file_ids"})
            references = self._string_list(arguments.get("file_ids"), "file_ids", 1, 8)
            file_ids, error = self._resolve_file_ids(references)
            if error:
                return self._encode_tool_result(error)
            by_id = {item.file_id: item for item in self._corpus.value.documents}
            payload = {
                "status": "ok",
                "documents": [
                    {
                        "file_id": file_id,
                        "file_name": by_id[file_id].file_name,
                        "overview": by_id[file_id].overview,
                    }
                    for file_id in file_ids
                ],
                "response_policy": "use_only_tool_data",
            }
            return self._encode_tool_result(payload)

        if call.name == "search_documents":
            self._require_exact_keys(arguments, {"queries", "file_ids", "limit"})
            queries = self._string_list(arguments.get("queries"), "queries", 1, 3)
            if any(len(query) > 500 for query in queries):
                raise ChatProtocolError("search query is too long")
            references = self._string_list(arguments.get("file_ids"), "file_ids", 1, 8)
            limit = arguments.get("limit")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 6:
                raise ChatProtocolError("search limit must be between 1 and 6")
            file_ids, error = self._resolve_file_ids(references)
            if error:
                return self._encode_tool_result(error)
            chunks = await self._rag.search(queries, file_ids, limit)
            if not chunks:
                return self._encode_tool_result(
                    {
                        "status": "no_usable_results",
                        "results": [],
                        "response_policy": "must_not_claim_document_facts",
                    }
                )
            return self._encode_tool_result(
                {
                    "status": "ok",
                    "results": [
                        {
                            "file_id": chunk.file_id,
                            "file_name": chunk.file_name,
                            "chunk_id": chunk.chunk_id,
                            "refs": chunk.refs,
                            "text": chunk.text,
                        }
                        for chunk in chunks
                    ],
                    "response_policy": "use_only_tool_data_and_add_citations",
                    "citation_required": True,
                }
            )

        raise ChatProtocolError(f"unknown tool: {call.name}")

    def _resolve_file_ids(
        self, references: list[str]
    ) -> tuple[list[str], dict[str, object] | None]:
        """Resolve file IDs or unique filenames into canonical IDs."""
        documents = self._corpus.value.documents
        by_id = {document.file_id: document for document in documents}
        resolved: list[str] = []
        for reference in references:
            if reference in by_id:
                file_id = reference
            else:
                matches = [
                    document.file_id
                    for document in documents
                    if document.file_name.casefold() == reference.casefold()
                ]
                if not matches:
                    return [], {
                        "status": "error",
                        "code": "unknown_file",
                        "reference": reference,
                        "available_files": [
                            {
                                "file_id": document.file_id,
                                "file_name": document.file_name,
                            }
                            for document in documents
                        ],
                        "response_policy": "ask_for_clarification",
                    }
                if len(matches) > 1:
                    return [], {
                        "status": "error",
                        "code": "ambiguous_file",
                        "reference": reference,
                        "matching_file_ids": matches,
                        "response_policy": "ask_for_clarification",
                    }
                file_id = matches[0]
            if file_id not in resolved:
                resolved.append(file_id)
        return resolved, None

    def _encode_tool_result(self, payload: dict[str, object]) -> str:
        """Serialize tool data, shrinking long text to the context budget."""
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= self._settings.max_context_chars:
            return json.dumps(payload, ensure_ascii=False, indent=2)

        text_fields: list[dict[str, Any]] = []
        for key in ("documents", "results"):
            values = payload.get(key)
            if isinstance(values, list):
                text_fields.extend(item for item in values if isinstance(item, dict))
        # Halving preserves every result's metadata before dropping useful text.
        while len(encoded) > self._settings.max_context_chars and text_fields:
            changed = False
            for item in text_fields:
                for key in ("overview", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and len(value) > 256:
                        item[key] = value[: max(256, len(value) // 2)]
                        changed = True
            payload["truncated"] = True
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if not changed:
                break
        if len(encoded) > self._settings.max_context_chars:
            raise ChatProtocolError("tool result exceeds the context limit")
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _persist_turn(self, user: str, assistant: str) -> None:
        """Atomically persist and expose one completed chat turn."""
        if not assistant.strip():
            raise ChatProtocolError("model returned an empty answer")
        candidate = self._history.value.with_turn(user, assistant)
        candidate.save(self._settings.history_path)
        self._history.value = candidate

    @staticmethod
    def _require_exact_keys(value: dict[str, object], expected: set[str]) -> None:
        """Reject tool arguments outside their declared schema."""
        if set(value) != expected:
            raise ChatProtocolError("tool arguments have unexpected or missing fields")

    @staticmethod
    def _string_list(
        value: object, label: str, minimum: int, maximum: int
    ) -> list[str]:
        """Validate, trim, and return a bounded string array."""
        if not isinstance(value, list) or not minimum <= len(value) <= maximum:
            raise ChatProtocolError(
                f"{label} must contain between {minimum} and {maximum} values"
            )
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ChatProtocolError(f"{label} must contain nonempty strings")
            cleaned.append(item.strip())
        return cleaned
