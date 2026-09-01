"""Validated async clients for local model endpoints and the Responses SDK."""

import asyncio
import json
import math
from typing import Any

import httpx
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from src.config import Settings
from src.models import Chunk


class ModelHTTPError(RuntimeError):
    """A bounded local-model transport or response-protocol failure."""


class LocalModelClients:
    """Call the local overview, embedding, and reranking model endpoints."""

    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient,
        *,
        embedding_gate: asyncio.Semaphore,
        rerank_gate: asyncio.Semaphore,
    ) -> None:
        self._settings = settings
        self._http = http
        self._embedding_gate = embedding_gate
        self._rerank_gate = rerank_gate

    async def complete_overview(self, file_name: str, chunks: list[Chunk]) -> str:
        """Generate the legacy-compatible bounded Vietnamese document overview."""
        context = "\n\n---\n\n".join(
            f"[{', '.join(chunk.refs)}]\n{chunk.text}" for chunk in chunks
        )[: self._settings.max_context_chars]
        payload = await self._request_json(
            "overview",
            f"{self._settings.llm_url}/v1/chat/completions",
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
                        "content": f"Tài liệu {file_name}:\n\n{context}",
                    },
                ],
                "stream": False,
                "max_tokens": 768,
                "temperature": 0.1,
            },
        )
        try:
            overview = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelHTTPError("invalid overview response") from exc
        if not isinstance(overview, str):
            raise ModelHTTPError("invalid overview response: missing content")
        result = overview.strip()
        if not result:
            raise ModelHTTPError("overview model returned empty content")
        return result

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one bounded batch and restore vectors to request order."""
        if not texts:
            return []
        payload = await self._request_json_gated(
            self._embedding_gate,
            "embedding",
            f"{self._settings.embed_url}/embedding",
            {"content": texts},
        )
        if not isinstance(payload, list) or len(payload) != len(texts):
            raise ModelHTTPError("invalid embedding response: wrong row count")

        vectors: list[list[float] | None] = [None] * len(texts)
        dimension: int | None = None
        for item in payload:
            if not isinstance(item, dict):
                raise ModelHTTPError("invalid embedding response item")
            index = self._response_index(item.get("index"), len(texts), "embedding")
            if vectors[index] is not None:
                raise ModelHTTPError("invalid embedding response: duplicate index")
            nested = item.get("embedding")
            if (
                not isinstance(nested, list)
                or len(nested) != 1
                or not isinstance(nested[0], list)
            ):
                raise ModelHTTPError("invalid embedding response: expected nested vector")
            vector = self._finite_vector(nested[0], "embedding")
            if not vector:
                raise ModelHTTPError("invalid embedding response: empty vector")
            if not any(vector):
                raise ModelHTTPError("invalid embedding response: zero vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ModelHTTPError("invalid embedding response: inconsistent dimensions")
            vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise ModelHTTPError("invalid embedding response: missing index")
        return [vector for vector in vectors if vector is not None]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Rerank one bounded candidate batch and restore its input order."""
        if not documents:
            return []
        payload = await self._request_json_gated(
            self._rerank_gate,
            "reranking",
            f"{self._settings.rerank_url}/reranking",
            {"query": query, "documents": documents},
        )
        if not isinstance(payload, dict):
            raise ModelHTTPError("invalid reranking response")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(documents):
            raise ModelHTTPError("invalid reranking response: wrong result count")

        scores: list[float | None] = [None] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                raise ModelHTTPError("invalid reranking response item")
            index = self._response_index(item.get("index"), len(documents), "reranking")
            if scores[index] is not None:
                raise ModelHTTPError("invalid reranking response: duplicate index")
            score = item.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ModelHTTPError("invalid reranking response: nonnumeric score")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ModelHTTPError("invalid reranking response: nonfinite score")
            scores[index] = numeric_score

        if any(score is None for score in scores):
            raise ModelHTTPError("invalid reranking response: missing index")
        return [score for score in scores if score is not None]

    async def _request_json_gated(
        self,
        gate: asyncio.Semaphore,
        label: str,
        url: str,
        payload: dict[str, object],
    ) -> Any:
        """Limit one upstream HTTP batch without serializing local validation."""
        async with gate:
            return await self._request_json(label, url, payload)

    async def _request_json(
        self, label: str, url: str, payload: dict[str, object]
    ) -> Any:
        try:
            response = await self._http.post(url, json=payload)
            await self._validate_status(response, label)
        except ModelHTTPError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelHTTPError(f"{label} request timed out") from exc
        except httpx.HTTPError as exc:
            raise ModelHTTPError(f"{label} request failed") from exc
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelHTTPError(f"invalid {label} JSON response") from exc

    @staticmethod
    async def _validate_status(response: httpx.Response, label: str) -> None:
        if response.is_success:
            return
        await response.aread()
        detail = " ".join(response.text.split())[:160]
        message = f"{label} service returned HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        raise ModelHTTPError(message)

    @staticmethod
    def _response_index(value: object, size: int, label: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= size
        ):
            raise ModelHTTPError(f"invalid {label} response index")
        return value

    @staticmethod
    def _finite_vector(value: list[object], label: str) -> list[float]:
        vector: list[float] = []
        for number in value:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ModelHTTPError(f"invalid {label} response: nonnumeric vector")
            converted = float(number)
            if not math.isfinite(converted):
                raise ModelHTTPError(f"invalid {label} response: nonfinite vector")
            vector.append(converted)
        return vector


def build_agent_model(settings: Settings, http: httpx.AsyncClient) -> OpenAIResponsesModel:
    """Build the SDK model against llama.cpp's OpenAI-compatible Responses API."""
    client = AsyncOpenAI(
        base_url=f"{settings.llm_url}/v1",
        api_key="local",
        http_client=http,
    )
    return OpenAIResponsesModel(settings.agent_model, client)
