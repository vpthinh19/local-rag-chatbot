# Task 6 report — bounded agent sessions

## RED evidence

- Added `tests/test_sessions.py` before `src/sessions.py` existed.
- `uv run pytest tests/test_sessions.py -q` → collection failed as expected with `ModuleNotFoundError: No module named 'src.sessions'` (1 collection error).

## GREEN / verification evidence

- `uv run pytest tests/test_sessions.py -q` → `8 passed in 1.25s`.
- `uv run pytest -q` → `257 passed, 7 skipped in 3.88s`.
- `git diff --check` → exit 0 with no whitespace errors.

## Changed files

- `src/sessions.py`: SDK `SQLiteSession` factory with the 48-item limit, durable session metadata, 80-character first-message titles, complete visible message-pair projection, pure whole-turn bounded context selection, and a transactional overlay that defers durable pops and additions until commit.
- `tests/test_sessions.py`: context boundary/current-tool retention, failed-turn rollback over pre-existing durable history, commit replay, metadata/title/message isolation, and targeted session deletion coverage.

## Commit

- `feat: add bounded agent sessions`

## Fix round 1 — fail-closed durable rewinds

### SDK inspection

- In the pinned Agents SDK 0.22.0, the public `Session` protocol exposes separate `add_items` and `pop_item` calls only.
- `Runner` persists current input with `add_items`; its conversation-retry rewind then calls `pop_item` for that just-saved suffix. With `TransactionalSession`, those items are still pending, so this normal retry path never mutates the durable delegate.
- `SQLiteSession` commits each public `pop_item` and `add_items` operation separately. No supported public API combines a durable pop with an append atomically.

### RED evidence

- Replaced the prior mixed-pop commit expectation with failure-injection coverage requiring an exception before either delegate mutation.
- Added an append-failure regression and a pending-input retry-pop regression.
- `uv run pytest tests/test_sessions.py -q` → `1 failed, 9 passed in 1.29s`: the old commit replayed the durable pop and appended the replacement instead of failing closed.

### GREEN / verification evidence

- `uv run pytest tests/test_sessions.py -q` → `10 passed in 1.23s`.
- `uv run pytest -q` → `259 passed, 7 skipped in 3.92s`.
- `git diff --check` → exit 0 with no whitespace errors.

### Changes

- `TransactionalSession.commit()` now rejects every overlay containing a requested durable pop before calling the delegate. Addition-only commits still issue exactly one `add_items` call.
- The class and commit docstrings make the public-API atomicity boundary explicit.
- Regression tests prove a mixed durable-pop-plus-addition commit leaves durable history unchanged, a failing append leaves prior history unchanged, discard remains no-op, and the SDK-style retry pop removes pending input only.
