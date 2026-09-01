# Task 5 report — migrate legacy RAG state

## RED evidence

- Added `tests/test_migration.py` before `src/migration.py` existed.
- `uv run pytest tests/test_migration.py -q` → collection failed as expected with `ModuleNotFoundError: No module named 'src.migration'` (1 collection error).
- Tightened the signature invalidation test to require a configured embedding signature. `uv run pytest tests/test_migration.py::test_signature_change_invalidates_ready_documents_once -q` → failed as expected with `AttributeError: 'Settings' object has no attribute 'embedding_signature'` (1 failed).

## GREEN / verification evidence

- `uv run pytest tests/test_migration.py tests/test_database.py tests/test_models.py -q` → `29 passed in 0.40s`.
- `uv run pytest -q` → `247 passed, 7 skipped in 3.85s`.
- `git diff --check` → exit 0 with no whitespace errors.

## Changed files

- `src/migration.py`: independent legacy corpus/history parsing, durable per-part and completion markers, source-copy preservation, Session-interface-only history import, and signature invalidation/reindex scheduling.
- `src/config.py`: explicit `EMBEDDING_SIGNATURE`-configurable identity (default `BAAI/bge-m3`), allowing a model swap at the same endpoint to trigger durable reindexing.
- `src/database.py`: schema version 2 active-reindex partial unique index, preventing duplicate queued/running reindex jobs.
- `src/models.py`: bounded immutable `MigrationReport`.
- `tests/test_migration.py`: idempotence, malformed-record isolation, missing-source failure, upload preservation, empty-session-only history migration, and one-time signature invalidation coverage.

## Concerns

- `uv run ruff check src/migration.py src/database.py src/models.py tests/test_migration.py` could not run because this environment has no `ruff` executable (`Failed to spawn: ruff`). The project’s full pytest suite and whitespace check passed.
