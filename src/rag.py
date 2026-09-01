"""Compact in-memory hybrid retrieval over persisted chunks.

BM25S uses a lowercase word-regex tokenizer here. It handles Vietnamese
diacritics, but it does not perform Vietnamese compound-word segmentation.
"""

import asyncio
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
import math
from typing import Protocol

import bm25s
import numpy as np

from src.llama import LlamaClient
from src.models import Chunk, Corpus, DocumentRecord, StoredChunk


@dataclass(frozen=True, slots=True)
class _IndexState:
    """An installable, internally aligned retrieval snapshot."""

    chunks: tuple[Chunk, ...]
    vectors: np.ndarray
    lexical: bm25s.BM25 | None


class RagIndex:
    """Combine BM25 and embeddings, then rerank a bounded candidate set."""

    def __init__(
        self,
        llama: LlamaClient,
        *,
        batch_size: int,
        lexical_limit: int,
        semantic_limit: int,
        candidate_limit: int,
        final_limit: int,
    ) -> None:
        """Configure bounded retrieval stages and an empty initial state."""
        limits = (
            batch_size,
            lexical_limit,
            semantic_limit,
            candidate_limit,
            final_limit,
        )
        if any(value <= 0 for value in limits):
            raise ValueError("RAG limits must be positive")
        self._llama = llama
        self._batch_size = batch_size
        self._lexical_limit = lexical_limit
        self._semantic_limit = semantic_limit
        self._candidate_limit = min(candidate_limit, 16)
        self._final_limit = min(final_limit, 6)
        self._state = self._make_state((), np.empty((0, 0), dtype=np.float32))

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._state.chunks)

    @property
    def vector_shape(self) -> tuple[int, ...]:
        """Expose the embedding matrix shape for diagnostics."""
        return self._state.vectors.shape

    @property
    def vector_dtype(self) -> np.dtype:
        """Expose the embedding matrix dtype for diagnostics."""
        return self._state.vectors.dtype

    @property
    def file_ids(self) -> set[str]:
        """Return document IDs represented in the current index."""
        return {chunk.file_id for chunk in self._state.chunks}

    async def rebuild(self, corpus: Corpus) -> None:
        """Replace the index from a complete persisted corpus."""
        chunks = tuple(corpus.chunks)
        vectors = await self._embed_batched([chunk.text for chunk in chunks])
        candidate = self._make_state(chunks, vectors)
        self.install(candidate)

    async def prepare_add(self, chunks: list[Chunk]) -> _IndexState:
        """Build an add candidate without mutating the live index."""
        if not chunks:
            return self._state
        existing_ids = {
            (chunk.file_id, chunk.chunk_id) for chunk in self._state.chunks
        }
        new_ids: set[tuple[str, int]] = set()
        for chunk in chunks:
            key = (chunk.file_id, chunk.chunk_id)
            if key in existing_ids or key in new_ids:
                raise ValueError(f"duplicate RAG chunk identifier: {key}")
            new_ids.add(key)

        new_vectors = await self._embed_batched([chunk.text for chunk in chunks])
        current_vectors = self._state.vectors
        if current_vectors.shape[1] and new_vectors.shape[1] != current_vectors.shape[1]:
            raise ValueError("embedding dimension changed while adding chunks")
        if current_vectors.shape[0]:
            vectors = np.vstack((current_vectors, new_vectors)).astype(
                np.float32, copy=False
            )
        else:
            vectors = new_vectors
        return self._make_state(self._state.chunks + tuple(chunks), vectors)

    def prepare_remove(self, file_id: str) -> _IndexState:
        """Build a removal candidate without mutating the live index."""
        keep = [
            index
            for index, chunk in enumerate(self._state.chunks)
            if chunk.file_id != file_id
        ]
        if len(keep) == len(self._state.chunks):
            return self._state
        chunks = tuple(self._state.chunks[index] for index in keep)
        dimension = self._state.vectors.shape[1]
        vectors = (
            self._state.vectors[keep].copy()
            if keep
            else np.empty((0, dimension), dtype=np.float32)
        )
        return self._make_state(chunks, vectors)

    def prepare_clear(self) -> _IndexState:
        """Build an empty candidate preserving the vector dimension."""
        dimension = self._state.vectors.shape[1]
        return self._make_state(
            (), np.empty((0, dimension), dtype=np.float32)
        )

    def install(self, candidate_state: _IndexState) -> None:
        """Atomically expose a fully prepared retrieval snapshot."""
        if not isinstance(candidate_state, _IndexState):
            raise TypeError("candidate_state was not prepared by RagIndex")
        self._state = candidate_state

    async def search(
        self, queries: list[str], file_ids: list[str], limit: int
    ) -> list[Chunk]:
        """Retrieve across query rewrites and return reranker-sorted chunks."""
        state = self._state
        clean_queries = [query.strip() for query in queries if query.strip()]
        if not state.chunks or not clean_queries or limit <= 0:
            return []

        selected_files = set(file_ids)
        allowed = np.asarray(
            [
                index
                for index, chunk in enumerate(state.chunks)
                if not selected_files or chunk.file_id in selected_files
            ],
            dtype=np.intp,
        )
        if allowed.size == 0:
            return []

        query_vectors = self._normalize_rows(
            await self._llama.embed(clean_queries),
            expected_rows=len(clean_queries),
            label="query embedding",
        )
        if query_vectors.shape[1] != state.vectors.shape[1]:
            raise ValueError("query embedding dimension does not match RAG index")

        # Query rewrites share candidates; retain each chunk's strongest rerank score.
        best_scores: dict[int, float] = {}
        for query, query_vector in zip(clean_queries, query_vectors, strict=True):
            lexical = self._lexical_ranking(state, query, allowed)
            semantic = self._semantic_ranking(state, query_vector, allowed)
            candidates = self._fuse(lexical, semantic)
            if not candidates:
                continue
            rerank_scores = await self._llama.rerank(
                query, [state.chunks[index].text for index in candidates]
            )
            if len(rerank_scores) != len(candidates):
                raise ValueError("reranker returned the wrong score count")
            for index, score in zip(candidates, rerank_scores, strict=True):
                numeric_score = float(score)
                if not math.isfinite(numeric_score):
                    raise ValueError("reranker returned a nonfinite score")
                best_scores[index] = max(
                    best_scores.get(index, -math.inf), numeric_score
                )

        result_limit = min(limit, self._final_limit, 6)
        ranked = sorted(best_scores, key=lambda index: (-best_scores[index], index))
        return [state.chunks[index] for index in ranked[:result_limit]]

    async def _embed_batched(self, texts: list[str]) -> np.ndarray:
        """Embed bounded batches into one normalized float32 matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        rows: list[list[float]] = []
        expected_dimension: int | None = None
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            values = await self._llama.embed(batch)
            matrix = self._normalize_rows(
                values,
                expected_rows=len(batch),
                label="document embedding",
            )
            if expected_dimension is None:
                expected_dimension = matrix.shape[1]
            elif matrix.shape[1] != expected_dimension:
                raise ValueError("embedding dimension changed between batches")
            rows.extend(matrix.tolist())
        return np.asarray(rows, dtype=np.float32)

    def _lexical_ranking(
        self, state: _IndexState, query: str, allowed: np.ndarray
    ) -> list[int]:
        """Rank allowed chunks by positive BM25 score."""
        if state.lexical is None:
            return []
        tokens = self._tokenize_one(query)
        if not tokens:
            return []
        scores = state.lexical.get_scores(tokens)
        eligible = [int(index) for index in allowed if scores[index] > 0]
        return sorted(eligible, key=lambda index: (-float(scores[index]), index))[
            : self._lexical_limit
        ]

    def _semantic_ranking(
        self, state: _IndexState, query_vector: np.ndarray, allowed: np.ndarray
    ) -> list[int]:
        """Rank allowed chunks by cosine similarity."""
        scores = state.vectors[allowed] @ query_vector
        pairs = zip(allowed.tolist(), scores.tolist(), strict=True)
        return [
            index
            for index, _ in sorted(pairs, key=lambda item: (-item[1], item[0]))[
                : self._semantic_limit
            ]
        ]

    def _fuse(self, *rankings: list[int]) -> list[int]:
        """Fuse lexical and semantic ranks with reciprocal rank fusion."""
        scores: dict[int, float] = {}
        # RRF compares ranks rather than incompatible BM25/cosine score scales.
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (60 + rank)
        return sorted(scores, key=lambda index: (-scores[index], index))[
            : self._candidate_limit
        ]

    @classmethod
    def _make_state(
        cls, chunks: tuple[Chunk, ...], vectors: np.ndarray
    ) -> _IndexState:
        """Validate aligned vectors and construct the matching BM25 index."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("RAG vectors must align with chunks")
        lexical: bm25s.BM25 | None = None
        if chunks:
            lexical = bm25s.BM25()
            lexical.index(
                cls._tokenize_many([chunk.text for chunk in chunks]),
                show_progress=False,
            )
        return _IndexState(chunks, vectors, lexical)

    @staticmethod
    def _normalize_rows(
        values: list[list[float]], *, expected_rows: int, label: str
    ) -> np.ndarray:
        """Validate and L2-normalize embedding rows for cosine products."""
        try:
            matrix = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {label} values") from exc
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] == 0:
            raise ValueError(f"invalid {label} shape")
        if not np.isfinite(matrix).all():
            raise ValueError(f"invalid {label}: nonfinite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(f"invalid {label}: zero vector")
        return (matrix / norms).astype(np.float32, copy=False)

    @staticmethod
    def _tokenize_many(texts: list[str]) -> list[list[str]]:
        """Tokenize text for BM25 without language-specific stemming."""
        return bm25s.tokenize(
            texts,
            stopwords=None,
            stemmer=None,
            return_ids=False,
            show_progress=False,
        )

    @classmethod
    def _tokenize_one(cls, text: str) -> list[str]:
        """Tokenize one query with the document tokenizer."""
        return cls._tokenize_many([text])[0]


class _ModelClients(Protocol):
    """The narrow retrieval-facing model boundary."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def rerank(self, query: str, documents: list[str]) -> list[float]: ...


class _PublicationLock:
    """An async lock that can prove its current task owns publication."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None

    async def __aenter__(self) -> "_PublicationLock":
        await self._lock.acquire()
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - asyncio always supplies a task here
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
        """Expose the standard lock diagnostic used by operational callers."""
        return self._lock.locked()


def _validate_snapshot_records(
    documents: tuple[DocumentRecord, ...], chunks: tuple[StoredChunk, ...]
) -> None:
    """Validate durable record membership and declared chunk counts."""
    if not all(isinstance(document, DocumentRecord) for document in documents):
        raise TypeError("snapshot documents must be DocumentRecord values")
    if not all(isinstance(chunk, StoredChunk) for chunk in chunks):
        raise TypeError("snapshot chunks must be StoredChunk values")
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


def _normalize_rows(
    values: object, *, expected_rows: int, label: str
) -> np.ndarray:
    """Validate and L2-normalize a complete vector matrix."""
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
    """An immutable, fully aligned retrieval view of ready durable records."""

    documents: tuple[DocumentRecord, ...]
    chunks: tuple[StoredChunk, ...]
    vectors: np.ndarray
    lexical: bm25s.BM25 | None

    def __post_init__(self) -> None:
        if not isinstance(self.documents, tuple) or not all(
            isinstance(document, DocumentRecord) for document in self.documents
        ):
            raise TypeError("snapshot documents must be a tuple of DocumentRecord values")
        if not isinstance(self.chunks, tuple) or not all(
            isinstance(chunk, StoredChunk) for chunk in self.chunks
        ):
            raise TypeError("snapshot chunks must be a tuple of StoredChunk values")
        _validate_snapshot_records(self.documents, self.chunks)
        detached = _normalize_rows(
            self.vectors,
            expected_rows=len(self.chunks),
            label="snapshot vector",
        )
        detached.setflags(write=False)
        object.__setattr__(self, "vectors", detached)

    @property
    def document_ids(self) -> frozenset[str]:
        """Return the ready document identifiers represented by this snapshot."""
        return frozenset(document.id for document in self.documents)


class SnapshotStore:
    """Publish complete snapshots only after the matching durable commit."""

    def __init__(self, initial: IndexSnapshot) -> None:
        if not isinstance(initial, IndexSnapshot):
            raise TypeError("initial snapshot must be an IndexSnapshot")
        self._current = initial
        self.publication_lock = _PublicationLock()

    async def capture(self) -> IndexSnapshot:
        """Capture one snapshot reference outside the database/install interval."""
        async with self.publication_lock:
            return self._current

    def install_locked(self, candidate: IndexSnapshot) -> None:
        """Install a prepared candidate while this task owns the publication gate."""
        if not isinstance(candidate, IndexSnapshot):
            raise TypeError("candidate snapshot must be an IndexSnapshot")
        if not self.publication_lock.held_by_current_task():
            raise RuntimeError("install_locked requires the publication lock")
        self._current = candidate


class RagService:
    """Build persisted-vector snapshots and retrieve against explicit snapshots."""

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
        """Configure bounded CPU and model-facing retrieval stages."""
        self._models = models
        self._cpu_executor = cpu_executor
        self._batch_size = self._pick_limit(
            "batch_size", batch_size, embedding_batch_size
        )
        self._lexical_limit = self._pick_limit(
            "lexical_limit", lexical_limit, lexical_candidate_limit
        )
        self._semantic_limit = self._pick_limit(
            "semantic_limit", semantic_limit, semantic_candidate_limit
        )
        self._candidate_limit = min(
            self._pick_limit("candidate_limit", candidate_limit, fused_candidate_limit), 16
        )
        self._final_limit = min(
            self._pick_limit("final_limit", final_limit, final_chunk_limit), 6
        )

    async def build(
        self,
        documents: Sequence[DocumentRecord],
        chunks: Sequence[StoredChunk],
        vectors: np.ndarray,
    ) -> IndexSnapshot:
        """Construct a snapshot from persisted vectors without embedding any text."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._cpu_executor,
            self._build_snapshot,
            tuple(documents),
            tuple(chunks),
            vectors,
        )

    async def search(
        self,
        snapshot: IndexSnapshot,
        queries: list[str],
        document_ids: list[str],
        limit: int,
    ) -> list[StoredChunk]:
        """Retrieve using one already-captured snapshot for the full request."""
        if not isinstance(snapshot, IndexSnapshot):
            raise TypeError("search requires an IndexSnapshot")
        clean_queries = [query.strip() for query in queries if query.strip()]
        if not snapshot.chunks or not clean_queries or limit <= 0:
            return []

        selected = set(document_ids)
        allowed = np.asarray(
            [
                index
                for index, chunk in enumerate(snapshot.chunks)
                if not selected or chunk.document_id in selected
            ],
            dtype=np.intp,
        )
        if allowed.size == 0:
            return []

        query_vectors = await self._embed_batched(clean_queries)
        if query_vectors.shape[1] != snapshot.vectors.shape[1]:
            raise ValueError("query embedding dimension does not match RAG index")

        best_scores: dict[int, float] = {}
        loop = asyncio.get_running_loop()
        for query, vector in zip(clean_queries, query_vectors, strict=True):
            candidates = await loop.run_in_executor(
                self._cpu_executor,
                self._rank_candidates,
                snapshot,
                query,
                vector,
                allowed,
            )
            if not candidates:
                continue
            scores = await self._models.rerank(
                query, [snapshot.chunks[index].text for index in candidates]
            )
            if len(scores) != len(candidates):
                raise ValueError("reranker returned the wrong score count")
            for index, score in zip(candidates, scores, strict=True):
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError("reranker returned a nonnumeric score")
                numeric = float(score)
                if not math.isfinite(numeric):
                    raise ValueError("reranker returned a nonfinite score")
                best_scores[index] = max(best_scores.get(index, -math.inf), numeric)

        result_limit = min(limit, self._final_limit, 6)
        ranked = sorted(best_scores, key=lambda index: (-best_scores[index], index))
        return [snapshot.chunks[index] for index in ranked[:result_limit]]

    async def _embed_batched(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        dimension: int | None = None
        loop = asyncio.get_running_loop()
        for start in range(0, len(texts), self._batch_size):
            values = await self._models.embed(texts[start : start + self._batch_size])
            matrix, dimension = await loop.run_in_executor(
                self._cpu_executor,
                partial(
                    self._prepare_query_batch,
                    values,
                    expected_rows=min(self._batch_size, len(texts) - start),
                    expected_dimension=dimension,
                ),
            )
            batches.append(matrix)
        return await loop.run_in_executor(
            self._cpu_executor, self._combine_query_batches, tuple(batches)
        )

    @staticmethod
    def _prepare_query_batch(
        values: object, *, expected_rows: int, expected_dimension: int | None
    ) -> tuple[np.ndarray, int]:
        """Normalize one model batch and validate it against prior batches."""
        matrix = RagService._normalize_rows(
            values, expected_rows=expected_rows, label="query embedding"
        )
        dimension = matrix.shape[1]
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError("embedding dimension changed between batches")
        return matrix, dimension

    @staticmethod
    def _combine_query_batches(batches: tuple[np.ndarray, ...]) -> np.ndarray:
        """Join normalized query batches without vector work on the event loop."""
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

    def _rank_candidates(
        self,
        snapshot: IndexSnapshot,
        query: str,
        query_vector: np.ndarray,
        allowed: np.ndarray,
    ) -> list[int]:
        lexical = self._lexical_ranking(snapshot, query, allowed)
        semantic = self._semantic_ranking(snapshot, query_vector, allowed)
        return self._fuse(lexical, semantic)

    def _lexical_ranking(
        self, snapshot: IndexSnapshot, query: str, allowed: np.ndarray
    ) -> list[int]:
        if snapshot.lexical is None:
            return []
        tokens = RagIndex._tokenize_one(query)
        if not tokens:
            return []
        scores = snapshot.lexical.get_scores(tokens)
        eligible = [int(index) for index in allowed if scores[index] > 0]
        return sorted(eligible, key=lambda index: (-float(scores[index]), index))[
            : self._lexical_limit
        ]

    def _semantic_ranking(
        self, snapshot: IndexSnapshot, vector: np.ndarray, allowed: np.ndarray
    ) -> list[int]:
        scores = snapshot.vectors[allowed] @ vector
        pairs = zip(allowed.tolist(), scores.tolist(), strict=True)
        return [
            index
            for index, _ in sorted(pairs, key=lambda item: (-item[1], item[0]))[
                : self._semantic_limit
            ]
        ]

    def _fuse(self, *rankings: list[int]) -> list[int]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                scores[index] = scores.get(index, 0.0) + 1.0 / (60 + rank)
        return sorted(scores, key=lambda index: (-scores[index], index))[ : self._candidate_limit]

    @staticmethod
    def _pick_limit(name: str, primary: int | None, alias: int | None) -> int:
        if primary is not None and alias is not None and primary != alias:
            raise ValueError(f"{name} was supplied twice with different values")
        value = primary if primary is not None else alias
        if value is None or value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _build_snapshot(
        documents: tuple[DocumentRecord, ...],
        chunks: tuple[StoredChunk, ...],
        vectors: np.ndarray,
    ) -> IndexSnapshot:
        _validate_snapshot_records(documents, chunks)
        lexical: bm25s.BM25 | None = None
        if chunks:
            lexical = bm25s.BM25()
            lexical.index(
                RagIndex._tokenize_many([chunk.text for chunk in chunks]),
                show_progress=False,
            )
        return IndexSnapshot(documents, chunks, vectors, lexical)

    @staticmethod
    def _normalize_rows(
        values: object, *, expected_rows: int, label: str
    ) -> np.ndarray:
        return _normalize_rows(values, expected_rows=expected_rows, label=label)
