"""Thin, validated HTTP client for the three llama.cpp servers."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
import math
from typing import Any

import httpx


class ModelHTTPError(RuntimeError):
    """A bounded model transport or response-protocol failure."""


@dataclass(frozen=True, slots=True)
class ContentEvent:
    """A streamed fragment of assistant-visible text."""

    content: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """One fully assembled tool call from the LLM stream."""

    index: int
    call_id: str
    name: str
    arguments: str


ChatEvent = ContentEvent | ToolCallEvent


@dataclass(slots=True)
class _ToolCallParts:
    """Fragments accumulated for one streamed tool call."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


class LlamaClient:
    """Validate the HTTP contracts of LLM, embed, and rerank servers."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        llm_url: str,
        embed_url: str,
        rerank_url: str,
    ) -> None:
        """Bind one shared HTTP client to the three model endpoints."""
        self._client = client
        self._llm_url = llm_url.rstrip("/")
        self._embed_url = embed_url.rstrip("/")
        self._rerank_url = rerank_url.rstrip("/")

    async def stream_chat(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        tool_choice: str | dict[str, object] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Stream text immediately and emit complete tool calls at EOF."""
        payload: dict[str, object] = {"messages": messages, "stream": True}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        # llama.cpp may split every tool-call field across multiple SSE deltas.
        tool_parts: dict[int, _ToolCallParts] = {}
        saw_done = False
        try:
            async with self._client.stream(
                "POST",
                f"{self._llm_url}/v1/chat/completions",
                json=payload,
            ) as response:
                await self._validate_status(response, "LLM stream")
                async for line in response.aiter_lines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith(":"):
                        continue
                    if not stripped.startswith("data:"):
                        continue
                    data = stripped[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    if not data:
                        continue

                    delta = self._parse_stream_delta(data)
                    content = delta.get("content")
                    if content is not None:
                        if not isinstance(content, str):
                            raise ModelHTTPError(
                                "invalid LLM stream: content must be text or null"
                            )
                        if content:
                            yield ContentEvent(content)

                    raw_tool_calls = delta.get("tool_calls")
                    if raw_tool_calls is not None:
                        self._accumulate_tool_calls(raw_tool_calls, tool_parts)

                if not saw_done:
                    raise ModelHTTPError("invalid LLM stream: missing [DONE]")

                for index in sorted(tool_parts):
                    parts = tool_parts[index]
                    if not parts.call_id or not parts.name or not parts.arguments:
                        raise ModelHTTPError(
                            "invalid LLM stream: incomplete tool call"
                        )
                    yield ToolCallEvent(
                        index=index,
                        call_id=parts.call_id,
                        name=parts.name,
                        arguments=parts.arguments,
                    )
        except ModelHTTPError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelHTTPError("LLM stream timed out") from exc
        except httpx.HTTPError as exc:
            raise ModelHTTPError("LLM stream request failed") from exc

    async def complete_chat(
        self,
        messages: list[dict[str, object]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Request one validated, non-streaming LLM completion."""
        payload = await self._request_json(
            "LLM completion",
            f"{self._llm_url}/v1/chat/completions",
            {
                "messages": messages,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelHTTPError("invalid LLM completion response") from exc
        if not isinstance(content, str):
            raise ModelHTTPError("invalid LLM completion response: missing content")
        return content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts and restore vectors to input order."""
        if not texts:
            return []
        payload = await self._request_json(
            "embedding",
            f"{self._embed_url}/embedding",
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
            if dimension is None:
                dimension = len(vector)
                if dimension == 0:
                    raise ModelHTTPError("invalid embedding response: empty vector")
            elif len(vector) != dimension:
                raise ModelHTTPError("invalid embedding response: inconsistent dimensions")
            vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise ModelHTTPError("invalid embedding response: missing index")
        return [vector for vector in vectors if vector is not None]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score documents and restore scores to input order."""
        if not documents:
            return []
        payload = await self._request_json(
            "reranking",
            f"{self._rerank_url}/reranking",
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
            index = self._response_index(
                item.get("index"), len(documents), "reranking"
            )
            if scores[index] is not None:
                raise ModelHTTPError("invalid reranking response: duplicate index")
            score = item.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ModelHTTPError("invalid reranking response: nonnumeric score")
            score = float(score)
            if not math.isfinite(score):
                raise ModelHTTPError("invalid reranking response: nonfinite score")
            scores[index] = score

        if any(score is None for score in scores):
            raise ModelHTTPError("invalid reranking response: missing index")
        return [score for score in scores if score is not None]

    async def _request_json(
        self, label: str, url: str, payload: dict[str, object]
    ) -> Any:
        """POST JSON and translate transport failures into one error type."""
        try:
            response = await self._client.post(url, json=payload)
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
        """Raise a bounded error containing the upstream response detail."""
        if response.is_success:
            return
        await response.aread()
        detail = " ".join(response.text.split())[:160]
        message = f"{label} service returned HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail}"
        raise ModelHTTPError(message)

    @staticmethod
    def _parse_stream_delta(data: str) -> dict[str, Any]:
        """Extract and validate an OpenAI-compatible stream delta."""
        try:
            payload = json.loads(data)
            choices = payload["choices"]
            choice = choices[0]
            delta = choice["delta"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelHTTPError("invalid LLM stream event") from exc
        if not isinstance(choices, list) or not isinstance(choice, dict):
            raise ModelHTTPError("invalid LLM stream choice")
        if not isinstance(delta, dict):
            raise ModelHTTPError("invalid LLM stream delta")
        return delta

    @staticmethod
    def _accumulate_tool_calls(
        raw_tool_calls: object, parts_by_index: dict[int, _ToolCallParts]
    ) -> None:
        """Append streamed tool-call fragments by their call index."""
        if not isinstance(raw_tool_calls, list):
            raise ModelHTTPError("invalid LLM stream: tool_calls must be an array")
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                raise ModelHTTPError("invalid LLM stream tool call")
            index = raw_call.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ModelHTTPError("invalid LLM stream tool call index")
            parts = parts_by_index.setdefault(index, _ToolCallParts())
            call_id = raw_call.get("id")
            if call_id is not None:
                if not isinstance(call_id, str):
                    raise ModelHTTPError("invalid LLM stream tool call id")
                parts.call_id += call_id
            function = raw_call.get("function")
            if function is not None:
                if not isinstance(function, dict):
                    raise ModelHTTPError("invalid LLM stream tool function")
                for key in ("name", "arguments"):
                    fragment = function.get(key)
                    if fragment is None:
                        continue
                    if not isinstance(fragment, str):
                        raise ModelHTTPError(
                            f"invalid LLM stream tool {key} fragment"
                        )
                    setattr(parts, key, getattr(parts, key) + fragment)

    @staticmethod
    def _response_index(value: object, size: int, label: str) -> int:
        """Validate an upstream result index against the request size."""
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
        """Convert a numeric vector while rejecting NaN and infinity."""
        vector: list[float] = []
        for number in value:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ModelHTTPError(f"invalid {label} response: nonnumeric vector")
            converted = float(number)
            if not math.isfinite(converted):
                raise ModelHTTPError(f"invalid {label} response: nonfinite vector")
            vector.append(converted)
        return vector
