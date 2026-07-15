import json
from pathlib import Path

import pytest

import src.models as models_module
from src.config import Settings
from src.models import (
    Chunk,
    Corpus,
    DataValidationError,
    Document,
    History,
    Message,
)


def _document(file_id: str = "doc-a", chunk_count: int = 1) -> Document:
    return Document(file_id, "quy-định.pdf", "Tóm tắt tiếng Việt", chunk_count)


def _chunk(file_id: str = "doc-a", chunk_id: int = 0) -> Chunk:
    return Chunk(
        file_id,
        "quy-định.pdf",
        chunk_id,
        ["p. 4", "pp. 4-5"],
        "Nội dung có dấu.",
    )


def test_corpus_round_trip_preserves_refs_and_unicode(tmp_path: Path) -> None:
    path = tmp_path / "corpus" / "corpus.json"
    corpus = Corpus(documents=[_document()], chunks=[_chunk()])

    corpus.save(path)

    assert Corpus.load(path) == corpus
    assert json.loads(path.read_text(encoding="utf-8"))["chunks"][0]["refs"] == [
        "p. 4",
        "pp. 4-5",
    ]


def test_legacy_summaries_are_migrated_to_documents(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "file_id": "old",
                        "file_name": "cũ.pdf",
                        "summary": "Tóm tắt cũ",
                        "chunk_count": 0,
                    }
                ],
                "chunks": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    corpus = Corpus.load(path)

    assert corpus.documents == [Document("old", "cũ.pdf", "Tóm tắt cũ", 0)]
    assert "documents" in corpus.to_dict()
    assert "summaries" not in corpus.to_dict()


def test_history_filters_legacy_internal_messages(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "legacy prompt"},
                    {"role": "user", "content": "Xin chào"},
                    {
                        "role": "assistant",
                        "content": "Chào bạn",
                        "rag_context": {"private": True},
                    },
                    {"role": "tool", "content": "internal result"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history = History.load(path)

    assert history.messages == [
        Message("user", "Xin chào"),
        Message("assistant", "Chào bạn"),
    ]
    assert history.to_dict() == {
        "messages": [
            {"role": "user", "content": "Xin chào"},
            {"role": "assistant", "content": "Chào bạn"},
        ]
    }


def test_missing_and_empty_files_load_as_empty_state(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    empty = tmp_path / "empty.json"
    empty.write_text("  \n", encoding="utf-8")

    assert Corpus.load(missing) == Corpus()
    assert Corpus.load(empty) == Corpus()
    assert History.load(missing) == History()
    assert History.load(empty) == History()


@pytest.mark.parametrize("raw", ["{", "[]", '{"chunks":"wrong"}'])
def test_malformed_corpus_has_a_clear_error(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(DataValidationError, match="corpus"):
        Corpus.load(path)


def test_serialization_failure_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.json"
    original = Corpus(documents=[_document(chunk_count=0)], chunks=[])
    original.save(path)
    before = path.read_bytes()

    def fail_serialization(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise TypeError("serialization failed")

    monkeypatch.setattr(models_module.json, "dumps", fail_serialization)

    with pytest.raises(TypeError, match="serialization failed"):
        Corpus().save(path)

    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_settings_paths_follow_the_configured_data_root(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "isolated")
    settings.ensure_dirs()

    paths = (
        settings.data_dir,
        settings.uploads_dir,
        settings.staging_dir,
        settings.corpus_path,
        settings.history_path,
    )
    assert all(path.is_relative_to(tmp_path) for path in paths)
    assert settings.uploads_dir.is_dir()
    assert settings.staging_dir.is_dir()
    assert settings.corpus_path.parent.is_dir()
    assert settings.history_path.parent.is_dir()


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "documents": [
                    _document(chunk_count=0).to_dict(),
                    _document(chunk_count=0).to_dict(),
                ],
                "chunks": [],
            },
            "duplicate document",
        ),
        (
            {
                "documents": [_document(chunk_count=0).to_dict()],
                "chunks": [_chunk().to_dict()],
            },
            "chunk_count",
        ),
        (
            {
                "documents": [_document(chunk_count=1).to_dict()],
                "chunks": [_chunk(file_id="unknown").to_dict()],
            },
            "unknown document",
        ),
    ],
)
def test_corpus_rejects_duplicate_or_mismatched_data(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        Corpus.from_dict(payload)


def test_corpus_helpers_return_new_values() -> None:
    empty = Corpus()
    populated = empty.with_document(_document(), [_chunk()])
    removed = populated.without_document("doc-a")

    assert empty == Corpus()
    assert populated.documents == [_document()]
    assert populated.chunks == [_chunk()]
    assert removed == Corpus()
