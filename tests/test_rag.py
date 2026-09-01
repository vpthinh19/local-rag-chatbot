import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.models import DocumentRecord, StoredChunk
from src.rag import IndexSnapshot, RagService, SnapshotStore


LIMITS = dict(batch_size=2, lexical_limit=4, semantic_limit=4, candidate_limit=16, final_limit=6)


def _document(identifier: str, count: int = 1) -> DocumentRecord:
    return DocumentRecord(identifier, f"{identifier}.pdf", "application/pdf", "ready", "", count, "", 1.0, 1.0)


def _chunk(identifier: str, text: str) -> StoredChunk:
    return StoredChunk(identifier, 0, ("p. 1",), text, None, None)


class _Models:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.embedded = False

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded = True
        self.started.set()
        await self.release.wait()
        return [self.vector for _ in texts]

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        return [1.0] * len(documents)


@pytest.mark.asyncio
async def test_search_keeps_its_captured_snapshot_during_publication() -> None:
    old = IndexSnapshot((_document("old"),), (_chunk("old", "alpha"),), np.array([[1.0, 0.0]], dtype=np.float32), None)
    new = IndexSnapshot((_document("new"),), (_chunk("new", "beta"),), np.array([[0.0, 1.0]], dtype=np.float32), None)
    store = SnapshotStore(old)
    models = _Models()
    with ThreadPoolExecutor(max_workers=2) as executor:
        rag = RagService(models, cpu_executor=executor, **LIMITS)
        task = asyncio.create_task(rag.search(await store.capture(), ["alpha"], ["old"], 1))
        await models.started.wait()
        async with store.publication_lock:
            store.install_locked(new)
        models.release.set()
        result = await task
    assert [chunk.document_id for chunk in result] == ["old"]


@pytest.mark.asyncio
async def test_build_from_persisted_vectors_never_embeds() -> None:
    models = _Models()
    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshot = await RagService(models, cpu_executor=executor, **LIMITS).build([_document("d")], [_chunk("d", "fact")], np.array([[3.0, 4.0]], dtype=np.float32))
    assert not models.embedded
    assert snapshot.vectors.tolist() == [[0.6000000238418579, 0.800000011920929]]


@pytest.mark.asyncio
async def test_large_snapshot_build_does_not_stall_the_event_loop() -> None:
    count = 5_000
    documents = [_document("d", count)]
    chunks = [StoredChunk("d", index, ("p. 1",), f"topic {index}", None, None) for index in range(count)]
    vectors = np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (count, 1))
    ticks = 0
    running = True

    async def ticker() -> None:
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    with ThreadPoolExecutor(max_workers=2) as executor:
        await RagService(_Models(), cpu_executor=executor, **LIMITS).build(documents, chunks, vectors)
    running = False
    await ticker_task
    assert ticks > 10
