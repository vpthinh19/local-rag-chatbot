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
