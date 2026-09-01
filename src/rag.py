"""Immutable, CPU-offloaded hybrid retrieval over persisted chunks."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
import math
from typing import Protocol

import bm25s
import numpy as np

from src.models import DocumentRecord, StoredChunk


class _ModelClients(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...


def _tokens(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(texts, stopwords=None, stemmer=None, return_ids=False, show_progress=False)


def _validate_records(documents: tuple[DocumentRecord, ...], chunks: tuple[StoredChunk, ...]) -> None:
    document_ids = {document.id for document in documents}
    if len(document_ids) != len(documents):
        raise ValueError("snapshot documents must have unique identifiers")
    counts = {document.id: 0 for document in documents}
    for chunk in chunks:
        if chunk.document_id not in counts:
            raise ValueError("snapshot chunk references an unknown document")
        counts[chunk.document_id] += 1
    if any(counts[document.id] != document.chunk_count for document in documents):
        raise ValueError("snapshot document chunk count mismatch")


def _normalize_rows(values: object, *, expected_rows: int, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {label} values") from exc
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(f"invalid {label} row count")
    if matrix.shape[1] == 0 and expected_rows:
        raise ValueError(f"invalid {label} dimension")
    if not np.isfinite(matrix).all():
        raise ValueError(f"invalid {label}: nonfinite values")
    if not expected_rows:
        return np.array(matrix, dtype=np.float32, copy=True, order="C")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"invalid {label}: zero vector")
    return np.asarray(matrix / norms, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """One complete aligned view, safe to retain for a full agent run."""

    documents: tuple[DocumentRecord, ...]
    chunks: tuple[StoredChunk, ...]
    vectors: np.ndarray
    lexical: bm25s.BM25 | None

    def __post_init__(self) -> None:
        if not isinstance(self.documents, tuple) or not all(isinstance(item, DocumentRecord) for item in self.documents):
            raise TypeError("snapshot documents must be a tuple of DocumentRecord values")
        if not isinstance(self.chunks, tuple) or not all(isinstance(item, StoredChunk) for item in self.chunks):
            raise TypeError("snapshot chunks must be a tuple of StoredChunk values")
        _validate_records(self.documents, self.chunks)
        vectors = _normalize_rows(self.vectors, expected_rows=len(self.chunks), label="snapshot vector")
        vectors.setflags(write=False)
        object.__setattr__(self, "vectors", vectors)

    @property
    def document_ids(self) -> frozenset[str]:
        return frozenset(document.id for document in self.documents)


class _PublicationLock:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None

    async def __aenter__(self) -> "_PublicationLock":
        await self._lock.acquire()
        task = asyncio.current_task()
        if task is None:  # pragma: no cover
            self._lock.release()
            raise RuntimeError("publication lock requires an asyncio task")
        self._owner = task
        return self

    async def __aexit__(self, *args: object) -> None:
        self._owner = None
        self._lock.release()

    def held_by_current_task(self) -> bool:
        return self._owner is asyncio.current_task()

    def locked(self) -> bool:
        return self._lock.locked()


class SnapshotStore:
    """Publish a candidate only while the matching database commit is gated."""

    def __init__(self, initial: IndexSnapshot) -> None:
        self._current = initial
        self.publication_lock = _PublicationLock()

    async def capture(self) -> IndexSnapshot:
        async with self.publication_lock:
            return self._current

    def install_locked(self, candidate: IndexSnapshot) -> None:
        if not isinstance(candidate, IndexSnapshot):
            raise TypeError("candidate snapshot must be an IndexSnapshot")
        if not self.publication_lock.held_by_current_task():
            raise RuntimeError("install_locked requires the publication lock")
        self._current = candidate


class RagService:
    """Build snapshots in the bounded CPU pool and search explicit snapshots."""

    def __init__(
        self,
        models: _ModelClients,
        *,
        cpu_executor: Executor,
        batch_size: int | None = None,
        lexical_limit: int | None = None,
        semantic_limit: int | None = None,
        candidate_limit: int | None = None,
        final_limit: int | None = None,
        embedding_batch_size: int | None = None,
        lexical_candidate_limit: int | None = None,
        semantic_candidate_limit: int | None = None,
        fused_candidate_limit: int | None = None,
        final_chunk_limit: int | None = None,
    ) -> None:
        self._models = models
        self._cpu_executor = cpu_executor
        self._batch_size = self._limit("batch_size", batch_size, embedding_batch_size)
        self._lexical_limit = self._limit("lexical_limit", lexical_limit, lexical_candidate_limit)
        self._semantic_limit = self._limit("semantic_limit", semantic_limit, semantic_candidate_limit)
        self._candidate_limit = min(self._limit("candidate_limit", candidate_limit, fused_candidate_limit), 16)
        self._final_limit = min(self._limit("final_limit", final_limit, final_chunk_limit), 6)

    async def build(self, documents: Sequence[DocumentRecord], chunks: Sequence[StoredChunk], vectors: np.ndarray) -> IndexSnapshot:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._cpu_executor, self._build_snapshot, tuple(documents), tuple(chunks), vectors)

    async def search(self, snapshot: IndexSnapshot, queries: list[str], document_ids: list[str], limit: int) -> list[StoredChunk]:
        if not isinstance(snapshot, IndexSnapshot):
            raise TypeError("search requires an IndexSnapshot")
        clean = [query.strip() for query in queries if query.strip()]
        if not snapshot.chunks or not clean or limit <= 0:
            return []
        selected = set(document_ids)
        allowed = np.asarray([index for index, chunk in enumerate(snapshot.chunks) if not selected or chunk.document_id in selected], dtype=np.intp)
        if not allowed.size:
            return []
        vectors = await self._embed_batched(clean)
        if vectors.shape[1] != snapshot.vectors.shape[1]:
            raise ValueError("query embedding dimension does not match RAG index")
        scores: dict[int, float] = {}
        loop = asyncio.get_running_loop()
        for query, vector in zip(clean, vectors, strict=True):
            candidates = await loop.run_in_executor(self._cpu_executor, self._rank, snapshot, query, vector, allowed)
            if not candidates:
                continue
            reranked = await self._models.rerank(query, [snapshot.chunks[index].text for index in candidates])
            if len(reranked) != len(candidates):
                raise ValueError("reranker returned the wrong score count")
            for index, score in zip(candidates, reranked, strict=True):
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                    raise ValueError("reranker returned a nonfinite score")
                scores[index] = max(scores.get(index, -math.inf), float(score))
        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        return [snapshot.chunks[index] for index in ranked[: min(limit, self._final_limit, 6)]]

    async def _embed_batched(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        dimension: int | None = None
        loop = asyncio.get_running_loop()
        for start in range(0, len(texts), self._batch_size):
            values = await self._models.embed(texts[start : start + self._batch_size])
            matrix, dimension = await loop.run_in_executor(
                self._cpu_executor,
                partial(self._prepare_batch, values, expected_rows=min(self._batch_size, len(texts) - start), expected_dimension=dimension),
            )
            batches.append(matrix)
        return await loop.run_in_executor(self._cpu_executor, np.concatenate, tuple(batches), 0)

    @staticmethod
    def _prepare_batch(values: object, *, expected_rows: int, expected_dimension: int | None) -> tuple[np.ndarray, int]:
        matrix = _normalize_rows(values, expected_rows=expected_rows, label="query embedding")
        if expected_dimension is not None and matrix.shape[1] != expected_dimension:
            raise ValueError("embedding dimension changed between batches")
        return matrix, matrix.shape[1]

    def _rank(self, snapshot: IndexSnapshot, query: str, vector: np.ndarray, allowed: np.ndarray) -> list[int]:
        lexical: list[int] = []
        if snapshot.lexical is not None:
            tokens = _tokens([query])[0]
            if tokens:
                values = snapshot.lexical.get_scores(tokens)
                lexical = sorted((int(index) for index in allowed if values[index] > 0), key=lambda index: (-float(values[index]), index))[: self._lexical_limit]
        semantic_scores = snapshot.vectors[allowed] @ vector
        semantic = [index for index, _ in sorted(zip(allowed.tolist(), semantic_scores.tolist(), strict=True), key=lambda item: (-item[1], item[0]))[: self._semantic_limit]]
        fused: dict[int, float] = {}
        for ranking in (lexical, semantic):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1 / (60 + rank)
        return sorted(fused, key=lambda index: (-fused[index], index))[: self._candidate_limit]

    @staticmethod
    def _build_snapshot(documents: tuple[DocumentRecord, ...], chunks: tuple[StoredChunk, ...], vectors: np.ndarray) -> IndexSnapshot:
        _validate_records(documents, chunks)
        lexical: bm25s.BM25 | None = None
        if chunks:
            lexical = bm25s.BM25()
            lexical.index(_tokens([chunk.text for chunk in chunks]), show_progress=False)
        return IndexSnapshot(documents, chunks, vectors, lexical)

    @staticmethod
    def _normalize_rows(values: object, *, expected_rows: int, label: str) -> np.ndarray:
        return _normalize_rows(values, expected_rows=expected_rows, label=label)

    @staticmethod
    def _limit(name: str, primary: int | None, alias: int | None) -> int:
        if primary is not None and alias is not None and primary != alias:
            raise ValueError(f"{name} was supplied twice with different values")
        value = primary if primary is not None else alias
        if value is None or value <= 0:
            raise ValueError(f"{name} must be positive")
        return value
