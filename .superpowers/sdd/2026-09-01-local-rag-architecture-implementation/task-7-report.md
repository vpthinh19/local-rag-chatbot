# Task 7 report — Agents SDK chat migration

## RED

- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q` before `src/agent.py` existed: `1 failed, 2 skipped, 6 errors`; every new Agent test failed because `src.agent` was missing.
- `uv run pytest tests/test_agent.py::test_stop_during_snapshot_capture_cancels_the_new_sdk_stream -q` before carrying a pending stop into the newly created SDK result: `1 failed`; the stream emitted `delta`, `done` instead of `cancelled`.

## GREEN

- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `11 passed, 2 skipped`.
- `uv run pytest -q`: `269 passed, 7 skipped`.

## Files

- Added `src/agent.py`: one shared SDK `Agent`, two strict read-only tools, immutable run context, bounded JSON serialization, application event translation, per-session run tracking, and cancellation-safe stop requests.
- Added `tests/test_agent.py`: snapshot isolation, unavailable-document rejection, schema bounds, bounded serialization, complete/cancelled transactional persistence, setup-race cancellation, and four-session concurrency coverage.

## Decisions

- Used the installed Agents SDK 0.22.0 public `ToolContext` to directly invoke decorated tools in tests. It carries the same `RunContextWrapper.context` specified by the design; a bare `RunContextWrapper` is not accepted by the pinned SDK’s public function-tool invoker.
- Tool functions read only `AgentContext.snapshot`; they never capture from `SnapshotStore`.
- A stop that arrives during snapshot capture is recorded and applied immediately after the single `Runner.run_streamed` call creates its stream.
- Errors and tool-status events are generic Vietnamese text and do not include prompts or document chunks.

## Concerns

- `stop()` and `stop_all()` request cancellation without waiting for a caller-paused async generator to settle. The stream’s `finally` block discards the transaction and clears the active-run entry when consumption resumes; Task 8’s SSE/disconnect boundary must continue consuming/settling the exact stream before deleting a session.

## Fix round 1 — SDK cancellation and live evaluation

### RED

- Updated the stream fake to match `RunResultStreaming.cancel()` by setting `is_complete=True`.
- `uv run pytest tests/test_agent.py::test_cancelled_sdk_stream_discards_the_turn tests/test_agent.py::test_stop_all_discards_every_real_sdk_shaped_cancelled_turn tests/test_agent.py::test_stop_during_snapshot_capture_cancels_the_new_sdk_stream -q`: `3 failed`. Each stopped stream incorrectly emitted `done`, demonstrating that SDK completion state alone cannot represent application success.

### GREEN

- `uv run pytest tests/test_agent.py::test_cancelled_sdk_stream_discards_the_turn tests/test_agent.py::test_stop_all_discards_every_real_sdk_shaped_cancelled_turn tests/test_agent.py::test_stop_during_snapshot_capture_cancels_the_new_sdk_stream tests/test_agent.py::test_disconnect_discards_a_partially_streamed_turn -q`: `4 passed`.
- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `13 passed, 2 skipped`.
- `uv run pytest -q`: `271 passed, 7 skipped`.

### Changes and decisions

- `src/agent.py` now records service-owned cancellation under the active-run lock before calling the SDK cancel method. Setup races, `stop_all`, SDK streams whose cancellation marks them complete, and the commit decision all consult that marker. A commit serializes against cancellation, so it either completes before a later stop observes no active run or a prior stop forces discard/cancelled.
- `tests/test_agent.py` makes the fake mirror the pinned SDK cancellation behavior and covers stop-all plus a client consumer closing after a delta.
- `tests/test_agent_eval.py` no longer imports `ChatAgent`, `LlamaClient`, or the custom two-completion loop. Its opt-in test builds the pinned Responses model, `LocalModelClients`, `RagService`, immutable snapshot, SQLite-backed `SessionService`, and `AgentService`; it observes persisted SDK function calls and final visible session messages through those real runtime boundaries.

## Fix round 2 — disconnected async-generator cleanup

### RED

- Replaced the disconnect double with a deterministic background-run fake that only settles after `cancel()` and uses a one-slot run gate.
- `uv run pytest tests/test_agent.py::test_disconnect_discards_a_partially_streamed_turn -q`: `1 failed`; after `aclose()`, `cancel_calls` was `0`, proving generator closure bypassed the prior `asyncio.CancelledError` cleanup.

### GREEN

- `uv run pytest tests/test_agent.py::test_disconnect_discards_a_partially_streamed_turn tests/test_agent.py::test_cancelled_sdk_stream_discards_the_turn tests/test_agent.py::test_stop_all_discards_every_real_sdk_shaped_cancelled_turn tests/test_agent.py::test_stop_during_snapshot_capture_cancels_the_new_sdk_stream -q`: `4 passed`.
- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `13 passed, 2 skipped`.
- `uv run pytest -q`: `271 passed, 7 skipped`.

### Changes and decisions

- Every uncommitted `AgentService.stream` exit now records service-owned cancellation and requests SDK cancellation from `finally`, including `aclose()`/disconnect closure. Completed, committed `done` runs skip this path.
- A per-active-run SDK-cancel marker ensures concurrent stop, stop-all, setup completion, outer task cancellation, and generator closure issue at most one SDK cancellation request; the service marker is set before that request.

## Fix round 3 — SDK drain and stale stop-all identities

### RED

- Replaced the disconnect fake with a drain-only SDK-faithful iterator: `cancel()` cancels a background task, while iterator settlement occurs only on a later `__anext__` during draining. Added a barrier-controlled completion double for stop-all identity reuse.
- `uv run pytest tests/test_agent.py::test_disconnect_discards_a_partially_streamed_turn tests/test_agent.py::test_stop_all_skips_a_stale_identity_that_committed_before_cancellation -q`: `2 failed`. `aclose()` returned before the iterator settled, and a delayed stale stop-all helper called `cancel()` after `done`.

### GREEN

- `uv run pytest tests/test_agent.py::test_disconnect_discards_a_partially_streamed_turn tests/test_agent.py::test_stop_all_skips_a_stale_identity_that_committed_before_cancellation -q`: `2 passed`.
- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `14 passed, 2 skipped`.
- `uv run pytest -q`: `272 passed, 7 skipped`.

### Changes and decisions

- Uncommitted cleanup now occurs inside the `_run_gate` scope. It records cancellation, calls `cancel()` once when needed, drains the saved `stream_events()` iterator without emitting translated events, then discards the transactional session; only then can the semaphore and active-session entry be released. Cleanup exceptions are suppressed so they cannot mask the original stream exit.
- Active runs now carry a completed marker. `stop`, `stop_all`, and cleanup cancellation requests verify the current `(session_id, active-object)` mapping under the lock; stale snapshot entries and already committed runs are not cancelled.
- Capturing cancellation state before cleanup preserves `error` for a genuine empty-final-output validation failure rather than relabeling it as `cancelled`.

## Fix round 4 — shielded terminal state and commit linearization

### RED

- Added deterministic cases for a queued late SDK delta, repeated task cancellation during drain and before gate acquisition, and cancellation during/before the transactional commit window.
- `uv run pytest tests/test_agent.py::test_stop_suppresses_a_queued_late_delta tests/test_agent.py::test_repeated_cancellation_during_drain_settles_before_propagating tests/test_agent.py::test_repeated_cancellation_before_gate_settles_active_run tests/test_agent.py::test_cancellation_after_commit_linearization_keeps_complete_turn tests/test_agent.py::test_cancellation_before_commit_linearization_discards_the_turn -q`: `4 failed, 1 passed`. The old service emitted the late delta, released terminal state on repeat cancellation, leaked pre-gate state, and lost an interrupted durable append.
- Added a lock-window regression where stop is queued after final-output validation but before commit linearization. `uv run pytest tests/test_agent.py::test_stop_winning_the_commit_lock_discards_without_done -q`: `1 failed`; the old branch emitted `done` after cancellation had won the commit lock.

### GREEN

- `uv run pytest tests/test_agent.py::test_repeated_cancellation_during_drain_settles_before_propagating -q`: `1 passed` after adding deferred cancellation propagation from generator-close cleanup.
- `uv run pytest tests/test_agent.py::test_stop_winning_the_commit_lock_discards_without_done tests/test_agent.py::test_stop_suppresses_a_queued_late_delta tests/test_agent.py::test_repeated_cancellation_during_drain_settles_before_propagating tests/test_agent.py::test_repeated_cancellation_before_gate_settles_active_run tests/test_agent.py::test_cancellation_after_commit_linearization_keeps_complete_turn tests/test_agent.py::test_cancellation_before_commit_linearization_discards_the_turn -q`: `6 passed`.
- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `20 passed, 2 skipped`.
- `uv run pytest -q`: `278 passed, 7 skipped`.

### State and technical ruling

- `_ActiveRun` now records `cancellation_requested`, `commit_started`, and `completed`. Cleanup and completed-turn finishing run in independent tasks awaited through a shield/retry loop; repeated caller cancellation is deferred until the iterator/task outcome, active-map removal, and settled event are complete.
- The event loop checks the service cancellation marker before translating each SDK event. Once marked, queued events are drained only; the caller receives no late delta.
- The commit linearization point is setting `commit_started` under `_active_lock` after stream completion and final-output validation. Before it, cancellation prevents commit and discards the overlay. After it, cancellation cannot request rollback; the single public `TransactionalSession.commit()` is shielded to a known outcome, then completion/removal is recorded under shield. This deliberately favors one complete durable turn over unsafe public-API compensation, because public SDK session writes cannot be atomically rolled back after a cancellation race.

## Fix round 5 — preserve the SDK iterator on caller cancellation

### RED

- Added a faithful iterator whose `__anext__()` becomes permanently closed if its owner task is cancelled, plus a snapshot-capture failure case. The direct caller-owned iteration path closes that iterator, making cleanup unable to drain the cancellation-resistant run loop before releasing the gate; setup capture exceptions previously produced EOF instead of an `error` event.

### GREEN

- `uv run pytest tests/test_agent.py::test_caller_cancellation_keeps_the_sdk_event_iterator_open_for_drain tests/test_agent.py::test_snapshot_capture_failure_yields_one_error_and_cleans_active -q`: `2 passed`.
- `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`: `22 passed, 2 skipped`.
- `uv run pytest -q`: `280 passed, 7 skipped`.

### Changes and decisions

- `AgentService.stream` now creates one pending `anext(event_iterator)` task at a time and awaits it with `asyncio.shield`. If the caller is cancelled while waiting, the service records/calls SDK cancellation, shield-waits that pending next to settle, then lets the existing in-gate cleanup drain the still-open iterator before releasing the gate or active entry.
- `stop`, `stop_all`, and cleanup now share the same identity-validated record-and-cancel helper; no new producer or orchestration layer was introduced.
- The outer setup-exception path records one stable `error` event after cleanup, so snapshot/session/pre-run failures cannot silently end the response. Error contents remain generic application events.
