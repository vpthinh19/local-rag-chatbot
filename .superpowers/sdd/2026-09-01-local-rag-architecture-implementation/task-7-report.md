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
