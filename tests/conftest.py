import pytest

from src.models import Chunk, Corpus, Document


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        documents=[
            Document("doc-a", "rules.pdf", "Quy định", 2),
            Document("doc-b", "guide.pdf", "Hướng dẫn", 2),
        ],
        chunks=[
            Chunk("doc-a", "rules.pdf", 0, ["p. 1"], "deadline submission"),
            Chunk("doc-a", "rules.pdf", 1, ["p. 2"], "grading policy"),
            Chunk("doc-b", "guide.pdf", 0, ["p. 1"], "installation guide"),
            Chunk("doc-b", "guide.pdf", 1, ["p. 2"], "troubleshooting steps"),
        ],
    )
