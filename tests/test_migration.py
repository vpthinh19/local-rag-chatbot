"""Legacy JSON import and embedding-signature recovery contracts."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile as shutil_copyfile
import time

import pytest
import pytest_asyncio

import src.migration as migration_module
from src.config import Settings
from src.database import Database
from src.documents import DocumentService
from src.migration import migrate_legacy


class MemorySession:
    """A Session-shaped in-memory double with observable durable items."""

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    async def get_items(self, limit: int | None = None) -> list[dict[str, str]]:
        return list(self.items if limit is None else self.items[-limit:])

    async def add_items(self, items: list[dict[str, str]]) -> None:
        self.items.extend(items)


class MigrationHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.settings = Settings(data_dir=tmp_path / "data")
        self.settings.ensure_dirs()
        self.database = Database(self.settings.database_path, 2_000)
        self.session = MemorySession()

    @property
    def corpus_json(self) -> Path:
        return self.settings.legacy_corpus_path

    @property
    def history_json(self) -> Path:
        return self.settings.legacy_history_path

    def write_corpus_one_document(self) -> None:
        self._legacy_source("doc-one", "rules.pdf", b"legacy source")
        self._write_corpus(
            [
                {
                    "file_id": "doc-one",
                    "file_name": "rules.pdf",
                    "overview": "Quy định cũ",
                    "chunk_count": 2,
                }
            ],
            [
                {
                    "file_id": "doc-one",
                    "file_name": "rules.pdf",
                    "chunk_id": 0,
                    "refs": ["p. 1"],
                    "text": "Nội dung thứ nhất",
                },
                {
                    "file_id": "doc-one",
                    "file_name": "rules.pdf",
                    "chunk_id": 1,
                    "refs": ["p. 2"],
                    "text": "Nội dung thứ hai",
                },
            ],
        )

    def write_history_two_turns(self) -> None:
        self.history_json.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Xin chào"},
                        {"role": "assistant", "content": "Chào bạn"},
                        {"role": "user", "content": "Quy định là gì?"},
                        {"role": "assistant", "content": "Ở trang một."},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def write_mixed_valid_and_invalid_corpus(self) -> None:
        self._legacy_source("valid", "valid.pdf", b"valid source")
        self._write_corpus(
            [
                {
                    "file_id": "valid",
                    "file_name": "valid.pdf",
                    "overview": "Hợp lệ",
                    "chunk_count": 1,
                },
                {
                    "file_id": "invalid",
                    "file_name": "invalid.pdf",
                    "overview": "Sai",
                    "chunk_count": "not a count",
                },
            ],
            [
                {
                    "file_id": "valid",
                    "file_name": "valid.pdf",
                    "chunk_id": 0,
                    "refs": ["p. 1"],
                    "text": "Một đoạn hợp lệ",
                }
            ],
        )

    async def run(self):
        return await migrate_legacy(
            self.settings, self.database, lambda _session_id: self.session
        )

    async def count(self, table: str) -> int:
        return await self.database.read(
            lambda conn: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        )

    async def sdk_items(self) -> list[dict[str, str]]:
        return await self.session.get_items()

    def _legacy_source(self, document_id: str, file_name: str, data: bytes) -> None:
        (self.settings.uploads_dir / f"{document_id}_{file_name}").write_bytes(data)

    def _write_corpus(self, documents: list[dict[str, object]], chunks: list[dict[str, object]]) -> None:
        self.corpus_json.write_text(
            json.dumps({"documents": documents, "chunks": chunks}, ensure_ascii=False),
            encoding="utf-8",
        )


@pytest_asyncio.fixture
async def migration_harness(tmp_path: Path) -> MigrationHarness:
    harness = MigrationHarness(tmp_path)
    await harness.database.initialize()
    return harness


@pytest.mark.asyncio
async def test_legacy_migration_is_idempotent(migration_harness: MigrationHarness) -> None:
    """Would fail if a completed import inserted the same records again."""
    migration_harness.write_corpus_one_document()
    migration_harness.write_history_two_turns()
    corpus_before = migration_harness.corpus_json.read_bytes()
    history_before = migration_harness.history_json.read_bytes()

    first = await migration_harness.run()
    second = await migration_harness.run()

    assert first.imported_documents == 1
    assert second.imported_documents == 0
    assert await migration_harness.count("documents") == 1
    assert await migration_harness.count("document_jobs") == 1
    assert len(await migration_harness.sdk_items()) == 4
    assert migration_harness.corpus_json.read_bytes() == corpus_before
    assert migration_harness.history_json.read_bytes() == history_before
    assert (migration_harness.settings.uploads_dir / "doc-one").read_bytes() == b"legacy source"


@pytest.mark.asyncio
async def test_reconcile_keeps_legacy_originals_after_valid_and_malformed_imports(
    migration_harness: MigrationHarness,
) -> None:
    migration_harness.write_corpus_one_document()
    malformed = migration_harness.settings.uploads_dir / "broken_broken.pdf"
    malformed.write_bytes(b"malformed legacy source")
    migration_harness._write_corpus(
        [
            {
                "file_id": "doc-one",
                "file_name": "rules.pdf",
                "overview": "Quy định cũ",
                "chunk_count": 2,
            },
            {"file_id": "broken", "file_name": "broken.pdf", "chunk_count": "bad"},
        ],
        [
            {"file_id": "doc-one", "file_name": "rules.pdf", "chunk_id": 0, "refs": ["p. 1"], "text": "Nội dung thứ nhất"},
            {"file_id": "doc-one", "file_name": "rules.pdf", "chunk_id": 1, "refs": ["p. 2"], "text": "Nội dung thứ hai"},
        ],
    )
    original = migration_harness.settings.uploads_dir / "doc-one_rules.pdf"
    before = {path: path.read_bytes() for path in (original, malformed)}

    await migration_harness.run()
    await DocumentService(migration_harness.settings, migration_harness.database).reconcile_files()

    assert await migration_harness.count("documents") == 1
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.asyncio
async def test_malformed_record_does_not_discard_valid_record(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if one invalid JSON record aborted the entire corpus import."""
    migration_harness.write_mixed_valid_and_invalid_corpus()

    report = await migration_harness.run()

    assert report.imported_documents == 1
    assert report.errors
    assert await migration_harness.count("documents") == 1


@pytest.mark.asyncio
async def test_missing_legacy_source_is_a_failed_document_without_a_job(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if an unavailable source entered the reindex queue."""
    migration_harness._write_corpus(
        [
            {
                "file_id": "missing",
                "file_name": "missing.pdf",
                "overview": "Không có tệp",
                "chunk_count": 1,
            }
        ],
        [
            {
                "file_id": "missing",
                "file_name": "missing.pdf",
                "chunk_id": 0,
                "refs": ["p. 1"],
                "text": "Đoạn văn vẫn được giữ",
            }
        ],
    )

    report = await migration_harness.run()
    document = await migration_harness.database.read(
        lambda conn: conn.execute(
            "SELECT status, error FROM documents WHERE id = 'missing'"
        ).fetchone()
    )

    assert report.imported_documents == 1
    assert document is not None
    assert document[0] == "failed"
    assert "legacy source file is missing" in document[1]
    assert await migration_harness.count("document_jobs") == 0


@pytest.mark.asyncio
async def test_migration_keeps_an_existing_new_destination(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if migration overwrote a committed durable upload."""
    migration_harness.write_corpus_one_document()
    destination = migration_harness.settings.uploads_dir / "doc-one"
    destination.write_bytes(b"existing durable source")

    await migration_harness.run()

    assert destination.read_bytes() == b"existing durable source"


@pytest.mark.asyncio
async def test_interrupted_source_copy_leaves_no_final_file_and_retries(
    migration_harness: MigrationHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Would fail if an interrupted copy made a partial destination look committed."""
    migration_harness.write_corpus_one_document()
    destination = migration_harness.settings.uploads_dir / "doc-one"

    def interrupted_copy(source: Path, target: Path) -> None:
        del source
        target.write_bytes(b"partial")
        raise OSError("copy interrupted")

    monkeypatch.setattr(migration_module.shutil, "copyfile", interrupted_copy)
    first = await migration_harness.run()

    assert first.imported_documents == 0
    assert not destination.exists()
    assert await migration_harness.count("documents") == 0
    assert await migration_harness.database.read(
        lambda conn: conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'legacy_import_v1_corpus'"
        ).fetchone()
    ) is None

    monkeypatch.setattr(migration_module.shutil, "copyfile", shutil_copyfile)
    second = await migration_harness.run()

    assert second.imported_documents == 1
    assert destination.read_bytes() == b"legacy source"


@pytest.mark.asyncio
async def test_malformed_history_records_are_reported_without_blocking_valid_messages(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if skipped history records vanished before the durable marker is set."""
    migration_harness.history_json.write_text(
        json.dumps(
            {
                "messages": [
                    None,
                    {"role": "system", "content": "legacy prompt"},
                    {"role": "user", "content": "Xin chào"},
                    {"role": "assistant", "content": "Chào bạn"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = await migration_harness.run()

    assert await migration_harness.sdk_items() == [
        {"role": "user", "content": "Xin chào"},
        {"role": "assistant", "content": "Chào bạn"},
    ]
    assert any("legacy message 0" in error for error in report.errors)
    assert any("legacy message 1" in error for error in report.errors)


@pytest.mark.asyncio
async def test_history_import_only_adds_to_an_empty_legacy_session(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if migration appended legacy messages to session history."""
    migration_harness.write_history_two_turns()
    migration_harness.session.items = [{"role": "user", "content": "new chat"}]

    await migration_harness.run()

    assert await migration_harness.sdk_items() == [{"role": "user", "content": "new chat"}]


@pytest.mark.asyncio
async def test_signature_change_invalidates_ready_documents_once(
    migration_harness: MigrationHarness,
) -> None:
    """Would fail if a changed embedding configuration left stale vectors searchable."""
    now = time.time()
    await migration_harness.database.write(
        lambda conn: (
            conn.execute(
                "INSERT INTO documents VALUES(?, ?, ?, 'ready', '', 1, '', ?, ?)",
                ("ready", "ready.pdf", "application/pdf", now, now),
            ),
            conn.execute(
                "INSERT INTO chunks VALUES(?, 0, ?, ?, ?, 2)",
                ("ready", '["p. 1"]', "indexed text", b"12345678"),
            ),
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('embedding_signature', 'old')"
            ),
        )
    )
    object.__setattr__(migration_harness.settings, "embedding_signature", "new-embed-model")

    await migration_harness.run()
    await migration_harness.run()

    document, chunk, jobs = await migration_harness.database.read(
        lambda conn: (
            conn.execute("SELECT status FROM documents WHERE id = 'ready'").fetchone(),
            conn.execute(
                "SELECT embedding, embedding_dim FROM chunks WHERE document_id = 'ready'"
            ).fetchone(),
            conn.execute(
                "SELECT count(*) FROM document_jobs WHERE document_id = 'ready' "
                "AND operation = 'reindex'"
            ).fetchone()[0],
        )
    )

    assert document == ("processing",)
    assert chunk == (None, None)
    assert jobs == 1
