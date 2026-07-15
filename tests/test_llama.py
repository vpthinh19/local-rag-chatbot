import json

import httpx
import pytest

from src.llama import ContentEvent, LlamaClient, ModelHTTPError, ToolCallEvent


def _client(http: httpx.AsyncClient) -> LlamaClient:
    return LlamaClient(http, "http://llm", "http://embed", "http://rerank")


@pytest.mark.asyncio
async def test_stream_chat_yields_content_and_ignores_boundary_deltas() -> None:
    body = "\n".join(
        (
            ": keep-alive",
            'data: {"choices":[{"delta":{"role":"assistant","content":null}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"Xin "}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"chào"},"finish_reason":"stop"}]}',
            "",
            "data: [DONE]",
            "",
        )
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://llm/v1/chat/completions")
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        events = [
            event
            async for event in _client(http).stream_chat(
                [{"role": "user", "content": "hello"}]
            )
        ]

    assert events == [ContentEvent("Xin "), ContentEvent("chào")]


@pytest.mark.asyncio
async def test_stream_chat_accumulates_fragmented_tool_calls_by_index() -> None:
    body = "\n\n".join(
        (
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call-","function":{"name":"query_","arguments":"{\\"que"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"1","function":{"name":"documents","arguments":"ries\\":[\\"x\\"]}"}}]},'
            '"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
            "",
        )
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        events = [event async for event in _client(http).stream_chat([], tools=[])]

    assert events == [
        ToolCallEvent(
            index=0,
            call_id="call-1",
            name="query_documents",
            arguments='{"queries":["x"]}',
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "data: not-json\n\ndata: [DONE]\n\n",
        'data: {"choices":[]}\n\ndata: [DONE]\n\n',
        'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
    ],
)
async def test_stream_chat_rejects_malformed_or_incomplete_sse(body: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="stream"):
            _ = [event async for event in _client(http).stream_chat([])]


@pytest.mark.asyncio
async def test_complete_chat_returns_validated_message_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload == {
            "messages": [{"role": "user", "content": "summarize"}],
            "stream": False,
            "max_tokens": 128,
            "temperature": 0.2,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Tóm tắt"}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await _client(http).complete_chat(
            [{"role": "user", "content": "summarize"}], 128, 0.2
        )

    assert result == "Tóm tắt"


@pytest.mark.asyncio
async def test_embed_maps_indexed_nested_vectors_to_input_order() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://embed/embedding")
        assert json.loads(request.content) == {"content": ["a", "b"]}
        return httpx.Response(
            200,
            json=[
                {"index": 1, "embedding": [[3, 4]]},
                {"index": 0, "embedding": [[1.0, 2.0]]},
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        vectors = await _client(http).embed(["a", "b"])

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [{"index": 0, "embedding": [[]]}],
        [
            {"index": 0, "embedding": [[1.0, 2.0]]},
            {"index": 1, "embedding": [[1.0]]},
        ],
        [{"index": 0, "embedding": [[float("nan")]]}],
        [
            {"index": 0, "embedding": [[1.0]]},
            {"index": 0, "embedding": [[2.0]]},
        ],
    ],
)
async def test_embed_rejects_malformed_vectors(payload: object) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )

    text_count = 2 if isinstance(payload, list) and len(payload) == 2 else 1
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="embedding"):
            await _client(http).embed(["x"] * text_count)


@pytest.mark.asyncio
async def test_rerank_maps_scores_to_document_indices() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("http://rerank/reranking")
        assert json.loads(request.content) == {
            "query": "q",
            "documents": ["first", "second"],
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.25},
                    {"index": 0, "relevance_score": 0.75},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        scores = await _client(http).rerank("q", ["first", "second"])

    assert scores == [0.75, 0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [{"index": 2, "relevance_score": 1.0}],
        [
            {"index": 0, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 2.0},
        ],
        [{"index": 0, "relevance_score": float("inf")}],
    ],
)
async def test_rerank_rejects_invalid_indices_or_scores(results: object) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=json.dumps({"results": results}),
            headers={"content-type": "application/json"},
        )

    document_count = 2 if isinstance(results, list) and len(results) == 2 else 1
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="rerank"):
            await _client(http).rerank("q", ["doc"] * document_count)


@pytest.mark.asyncio
async def test_empty_model_inputs_do_not_make_requests() -> None:
    request_count = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = _client(http)
        assert await client.embed([]) == []
        assert await client.rerank("q", []) == []

    assert request_count == 0


@pytest.mark.asyncio
async def test_non_2xx_error_is_bounded() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="secret-" + "x" * 1_000)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError) as caught:
            await _client(http).embed(["x"])

    assert "503" in str(caught.value)
    assert len(str(caught.value)) < 300


@pytest.mark.asyncio
async def test_timeout_is_translated() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="timed out"):
            await _client(http).rerank("q", ["doc"])


class _CloseAwareStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"second"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_closing_stream_consumer_closes_response_not_shared_client() -> None:
    stream = _CloseAwareStream()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        iterator = _client(http).stream_chat([])
        assert await anext(iterator) == ContentEvent("first")
        await iterator.aclose()

        assert stream.closed is True
        assert http.is_closed is False
