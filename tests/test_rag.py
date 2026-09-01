import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.models import Chunk, Corpus, Document, DocumentRecord, StoredChunk
from src.rag import IndexSnapshot, RagIndex, RagService, SnapshotStore


class FakeLlama:
    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.embed_calls: list[list[str]] = []
        self.rerank_calls: list[tuple[str, list[str]]] = []
        self.rerank_scores: dict[tuple[str, str], float] = {}
        self.rerank_error: Exception | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self.vectors.get(text, [1.0, 1.0]) for text in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.rerank_calls.append((query, list(documents)))
        if self.rerank_error:
            raise self.rerank_error
        return [self.rerank_scores.get((query, document), 0.0) for document in documents]


def _index(llama: FakeLlama, **overrides: int) -> RagIndex:
    values = {
        "batch_size": 2,
        "lexical_limit": 4,
        "semantic_limit": 4,
        "candidate_limit": 16,
        "final_limit": 6,
    }
    values.update(overrides)
    return RagIndex(llama, **values)


@pytest.mark.asyncio
async def test_empty_corpus_rebuild_needs_no_model_request() -> None:
    llama = FakeLlama()
    index = _index(llama)

    await index.rebuild(Corpus())

    assert index.chunk_count == 0
    assert index.vector_shape == (0, 0)
    assert await index.search(["anything"], [], 3) == []
    assert llama.embed_calls == []


@pytest.mark.asyncio
async def test_rebuild_embeds_in_bounded_batches(corpus: Corpus) -> None:
    llama = FakeLlama()
    index = _index(llama, batch_size=2)

    await index.rebuild(corpus)

    assert [len(batch) for batch in llama.embed_calls] == [2, 2]
    assert index.chunk_count == 4
    assert index.vector_shape == (4, 2)
    assert index.vector_dtype == np.dtype(np.float32)


@pytest.mark.asyncio
async def test_prepare_add_embeds_only_new_chunks_and_waits_for_install(
    corpus: Corpus,
) -> None:
    llama = FakeLlama()
    index = _index(llama)
    await index.rebuild(corpus)
    llama.embed_calls.clear()
    chunk = Chunk("doc-c", "new.pdf", 0, ["p. 1"], "new content")

    candidate = await index.prepare_add([chunk])

    assert llama.embed_calls == [["new content"]]
    assert index.chunk_count == 4
    index.install(candidate)
    assert index.chunk_count == 5


@pytest.mark.asyncio
async def test_prepare_remove_does_not_mutate_live_state(corpus: Corpus) -> None:
    index = _index(FakeLlama())
    await index.rebuild(corpus)

    candidate = index.prepare_remove("doc-a")

    assert index.chunk_count == 4
    index.install(candidate)
    assert index.chunk_count == 2
    assert index.file_ids == {"doc-b"}


@pytest.mark.asyncio
async def test_prepare_clear_waits_for_install(corpus: Corpus) -> None:
    index = _index(FakeLlama())
    await index.rebuild(corpus)

    candidate = index.prepare_clear()

    assert index.chunk_count == 4
    index.install(candidate)
    assert index.chunk_count == 0
    assert index.vector_shape == (0, 2)


@pytest.mark.asyncio
async def test_file_filter_is_applied_before_reranking(corpus: Corpus) -> None:
    llama = FakeLlama()
    index = _index(llama)
    await index.rebuild(corpus)

    result = await index.search(["deadline"], ["doc-b"], 6)

    assert result
    assert {chunk.file_id for chunk in result} == {"doc-b"}
    assert llama.rerank_calls
    assert set(llama.rerank_calls[0][1]) <= {
        "installation guide",
        "troubleshooting steps",
    }


@pytest.mark.asyncio
async def test_lexical_and_semantic_rankings_are_fused() -> None:
    chunks = [
        Chunk("doc", "one.pdf", 0, [], "deadline policy"),
        Chunk("doc", "one.pdf", 1, [], "unrelated semantic target"),
    ]
    corpus = Corpus([Document("doc", "one.pdf", "", 2)], chunks)
    llama = FakeLlama(
        {
            "deadline policy": [1.0, 0.0],
            "unrelated semantic target": [0.0, 1.0],
            "deadline": [0.0, 1.0],
        }
    )
    index = _index(llama, lexical_limit=1, semantic_limit=1, candidate_limit=2)
    await index.rebuild(corpus)

    await index.search(["deadline"], [], 2)

    assert set(llama.rerank_calls[0][1]) == {
        "deadline policy",
        "unrelated semantic target",
    }


@pytest.mark.asyncio
async def test_multi_query_uses_one_embedding_call_and_best_rerank_score(
    corpus: Corpus,
) -> None:
    llama = FakeLlama()
    llama.rerank_scores = {
        ("q1", "deadline submission"): 2.0,
        ("q2", "deadline submission"): 8.0,
        ("q1", "grading policy"): 6.0,
        ("q2", "grading policy"): 1.0,
    }
    index = _index(llama)
    await index.rebuild(corpus)
    llama.embed_calls.clear()

    result = await index.search(["q1", "q2"], ["doc-a"], 2)

    assert llama.embed_calls == [["q1", "q2"]]
    assert result[0].text == "deadline submission"
    assert {chunk.text for chunk in result} == {
        "deadline submission",
        "grading policy",
    }


@pytest.mark.asyncio
async def test_candidate_and_final_limits_are_enforced() -> None:
    chunks = [
        Chunk("doc", "many.pdf", index, [], f"common text {index}")
        for index in range(20)
    ]
    corpus = Corpus([Document("doc", "many.pdf", "", 20)], chunks)
    llama = FakeLlama()
    index = _index(
        llama,
        lexical_limit=20,
        semantic_limit=20,
        candidate_limit=16,
        final_limit=6,
    )
    await index.rebuild(corpus)

    result = await index.search(["common"], [], 99)

    assert len(llama.rerank_calls[0][1]) == 16
    assert len(result) == 6


@pytest.mark.asyncio
async def test_zero_or_mismatched_vectors_leave_live_state_unchanged(
    corpus: Corpus,
) -> None:
    llama = FakeLlama()
    index = _index(llama)
    await index.rebuild(corpus)
    original_shape = index.vector_shape

    llama.vectors["new"] = [0.0, 0.0]
    zero = Chunk("new", "new.pdf", 0, [], "new")
    with pytest.raises(ValueError, match="zero"):
        await index.prepare_add([zero])
    assert index.chunk_count == 4
    assert index.vector_shape == original_shape

    llama.vectors["wide"] = [1.0, 0.0, 0.0]
    wide = Chunk("wide", "wide.pdf", 0, [], "wide")
    with pytest.raises(ValueError, match="dimension"):
        await index.prepare_add([wide])
    assert index.chunk_count == 4


@pytest.mark.asyncio
async def test_query_dimension_mismatch_is_rejected(corpus: Corpus) -> None:
    llama = FakeLlama({"query": [1.0, 0.0, 0.0]})
    index = _index(llama)
    await index.rebuild(corpus)

    with pytest.raises(ValueError, match="dimension"):
        await index.search(["query"], [], 2)


@pytest.mark.asyncio
async def test_reranker_failure_does_not_change_live_index(corpus: Corpus) -> None:
    llama = FakeLlama()
    index = _index(llama)
    await index.rebuild(corpus)
    before = (index.chunk_count, index.vector_shape, index.file_ids)
    llama.rerank_error = RuntimeError("reranker unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        await index.search(["deadline"], [], 2)

    assert (index.chunk_count, index.vector_shape, index.file_ids) == before


LIMITS = {
    "batch_size": 2,
    "lexical_limit": 4,
    "semantic_limit": 4,
    "candidate_limit": 16,
    "final_limit": 6,
}


def _record(document_id: str, *, chunk_count: int = 1) -> DocumentRecord:
    return DocumentRecord(
        document_id, f"{document_id}.pdf", "application/pdf", "ready", "", chunk_count, "", 1.0, 1.0
    )


def _stored_chunk(document_id: str, text: str) -> StoredChunk:
    return StoredChunk(document_id, 0, ("p. 1",), text, None, None)


def _snapshot_with(document_id: str, text: str, vector: list[float]) -> IndexSnapshot:
    matrix = np.asarray([vector], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix.setflags(write=False)
    lexical = RagIndex._make_state(  # noqa: SLF001 - reuse the existing tokenizer only
        (Chunk(document_id, f"{document_id}.pdf", 0, ["p. 1"], text),), matrix
    ).lexical
    return IndexSnapshot((_record(document_id),), (_stored_chunk(document_id, text),), matrix, lexical)


class _PausingModels:
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector
        self.embedding_started = asyncio.Event()
        self.release_embedding = asyncio.Event()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedding_started.set()
        await self.release_embedding.wait()
        return [self.query_vector for _ in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 for _ in documents]


class _FailingIfEmbeddedModels:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("persisted vectors must not be embedded")

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [1.0 for _ in documents]


@pytest.mark.asyncio
async def test_search_keeps_the_snapshot_it_started_with() -> None:
    old = _snapshot_with("old", "alpha fact", [1.0, 0.0])
    new = _snapshot_with("new", "beta fact", [0.0, 1.0])
    store = SnapshotStore(old)
    models = _PausingModels(query_vector=[1.0, 0.0])
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag = RagService(models, cpu_executor=executor, **LIMITS)
        task = asyncio.create_task(
            rag.search(await store.capture(), ["alpha"], ["old"], 5)
        )
        await models.embedding_started.wait()
        async with store.publication_lock:
            store.install_locked(new)
        models.release_embedding.set()
        result = await task

    assert [chunk.document_id for chunk in result] == ["old"]


@pytest.mark.asyncio
async def test_build_from_persisted_vectors_never_embeds() -> None:
    models = _FailingIfEmbeddedModels()
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag = RagService(models, cpu_executor=executor, **LIMITS)
        snapshot = await rag.build(
            [_record("document")],
            [_stored_chunk("document", "alpha fact")],
            np.array([[3.0, 4.0]], dtype=np.float32),
        )

    assert snapshot.vectors.tolist() == [[0.6000000238418579, 0.800000011920929]]


@pytest.mark.asyncio
async def test_snapshot_store_requires_the_publication_lock_to_install() -> None:
    snapshot = _snapshot_with("document", "alpha fact", [1.0, 0.0])
    store = SnapshotStore(snapshot)

    with pytest.raises(RuntimeError, match="publication lock"):
        store.install_locked(snapshot)

    async with store.publication_lock:
        store.install_locked(snapshot)
    assert await store.capture() is snapshot


@pytest.mark.asyncio
async def test_build_detaches_tuple_data_and_makes_normalized_vectors_read_only() -> None:
    document = _record("document")
    chunk = _stored_chunk("document", "alpha fact")
    source = np.array([[3.0, 4.0]], dtype=np.float32)
    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot = await RagService(
            _FailingIfEmbeddedModels(), cpu_executor=executor, **LIMITS
        ).build([document], [chunk], source)

    source[0, 0] = 99.0
    assert isinstance(snapshot.documents, tuple)
    assert isinstance(snapshot.chunks, tuple)
    assert snapshot.vectors.flags.writeable is False
    assert snapshot.vectors.tolist() == [[0.6000000238418579, 0.800000011920929]]
    with pytest.raises(ValueError):
        snapshot.vectors[0, 0] = 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors, error",
    [
        ([[1.0, 0.0], [0.0, 1.0]], "row count"),
        ([[0.0, 0.0]], "zero"),
        ([[float("nan"), 1.0]], "nonfinite"),
    ],
)
async def test_search_rejects_invalid_embedding_protocol(
    vectors: list[list[float]], error: str
) -> None:
    class InvalidModels(_FailingIfEmbeddedModels):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return vectors

    with ThreadPoolExecutor(max_workers=2) as executor:
        rag = RagService(InvalidModels(), cpu_executor=executor, **LIMITS)
        with pytest.raises(ValueError, match=error):
            await rag.search(_snapshot_with("document", "alpha fact", [1.0, 0.0]), ["alpha"], [], 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("scores, error", [([], "score count"), ([float("inf")], "nonfinite")])
async def test_search_rejects_invalid_reranker_protocol(
    scores: list[float], error: str
) -> None:
    class InvalidModels(_FailingIfEmbeddedModels):
        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        async def rerank(self, query: str, documents: list[str]) -> list[float]:
            return scores

    with ThreadPoolExecutor(max_workers=2) as executor:
        rag = RagService(InvalidModels(), cpu_executor=executor, **LIMITS)
        with pytest.raises(ValueError, match=error):
            await rag.search(_snapshot_with("document", "alpha fact", [1.0, 0.0]), ["alpha"], [], 1)
