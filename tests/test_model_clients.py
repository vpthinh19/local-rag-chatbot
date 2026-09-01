import asyncio
import json

import httpx
import pytest

from src.config import Settings
from src.model_clients import LocalModelClients, ModelHTTPError, build_agent_model
from src.models import Chunk


def _clients(http: httpx.AsyncClient) -> LocalModelClients:
    return LocalModelClients(
        Settings(max_context_chars=20),
        http,
        embedding_gate=asyncio.Semaphore(1),
        rerank_gate=asyncio.Semaphore(1),
    )


@pytest.mark.asyncio
async def test_complete_overview_uses_the_existing_bounded_context_format() -> None:
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": " Tóm tắt "}}]},
        )

    chunks = [
        Chunk("d", "report.pdf", 0, ["p. 1"], "first chunk"),
        Chunk("d", "report.pdf", 1, ["p. 2"], "second chunk"),
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        overview = await _clients(http).complete_overview("report.pdf", chunks)

    assert overview == "Tóm tắt"
    assert requests == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tạo overview tiếng Việt ngắn gọn cho tài liệu: tóm tắt, "
                        "dàn ý và các điểm chính, tối đa 300 từ. "
                        "Chỉ dùng nội dung được cung cấp."
                    ),
                },
                {
                    "role": "user",
                    "content": "Tài liệu report.pdf:\n\n[p. 1]\nfirst chunk\n\n",
                },
            ],
            "stream": False,
            "max_tokens": 768,
            "temperature": 0.1,
        }
    ]


@pytest.mark.asyncio
async def test_complete_overview_rejects_empty_model_content() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="empty"):
            await _clients(http).complete_overview(
                "report.pdf", [Chunk("d", "report.pdf", 0, [], "text")]
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "count"),
    [
        ([{"index": 0, "embedding": [[1.0, 2.0]]}], 2),
        (
            [
                {"index": 0, "embedding": [[1.0, 2.0]]},
                {"index": 1, "embedding": [[1.0]]},
            ],
            2,
        ),
        ([{"index": 0, "embedding": [[float("nan")]]}], 1),
        ([{"index": 0, "embedding": [[float("inf")]]}], 1),
        ([{"index": 0, "embedding": [[0.0, 0.0]]}], 1),
    ],
)
async def test_embed_rejects_invalid_protocol_vectors(payload: object, count: int) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(payload))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="embedding"):
            await _clients(http).embed(["text"] * count)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "results",
    [
        [],
        [{"index": 0, "relevance_score": float("nan")}],
        [{"index": 0, "relevance_score": float("inf")}],
    ],
)
async def test_rerank_rejects_invalid_protocol_scores(results: object) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"results": results}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ModelHTTPError, match="reranking"):
            await _clients(http).rerank("query", ["document"])


def test_build_agent_model_uses_the_local_responses_endpoint() -> None:
    settings = Settings(llm_url="http://llm")
    http = httpx.AsyncClient()
    try:
        model = build_agent_model(settings, http)
        assert model.model == "local"
        assert str(model._client.base_url) == "http://llm/v1/"  # noqa: SLF001
    finally:
        asyncio.run(http.aclose())
