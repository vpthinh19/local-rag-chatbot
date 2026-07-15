import numpy as np
import pytest

from src.models import Chunk, Corpus, Document
from src.rag import RagIndex


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
