# Local RAG Mini-Agent Implementation Plan

## Goal

Replace the legacy backend with a small FastAPI RAG agent that calls three persistent llama.cpp HTTP servers and spawns one disposable Docling subprocess per upload. Preserve the existing browser experience and JSON data while deleting the old model/service/cancellation architecture only after equivalent tests pass.

## Implementation principles

- Treat legacy code as contract/reference, not code to port class by class.
- Keep one Uvicorn worker and one active chat pipeline.
- Keep all persistent application state in FastAPI; isolate only Docling conversion/chunking.
- Direct LLM answer or at most one read-only tool call per user turn.
- Build candidate state before commit; cancel discards only uncommitted work.
- Add no framework, queue, database, vector store, or generic abstraction without a demonstrated need.
- Preserve current user changes to `test.py`, `test.txt`, `test.json`, `pyproject.toml`, `uv.lock`, and the README deletion unless a task explicitly brings a file into scope.
- Use concise commits and the repository's configured human identity when commits are requested.

## Target structure

```text
src/
  main.py
  config.py
  models.py
  llama.py
  rag.py
  docling_worker.py
  documents.py
  chat.py
  templates/index.html
  static/style.css
  static/script.js
tests/
  fixtures/agent_cases.json
  helpers/fake_docling_worker.py
  test_models.py
  test_llama.py
  test_rag.py
  test_docling_worker.py
  test_documents.py
  test_chat.py
  test_api.py
  test_ui_assets.py
  test_agent_eval.py
```

The exact number of helper functions is implementation-driven. Do not create extra production modules merely to mirror test files.

---

## Task 1: Establish the test harness and dependency boundary

### Files

- Modify `pyproject.toml`
- Modify `uv.lock`
- Create `tests/test_ui_assets.py`

### Work

1. Add direct runtime dependencies used by production code:
   - `bm25s`
   - `docling`
   - `fastapi[standard]`
   - `httpx`
   - `numpy`
   - `torch`
2. Add a dev dependency group containing `pytest` and `pytest-asyncio`.
3. Confirm `llama-cpp-python`, `model2vec`, OpenAI SDK, agent frameworks, task queues, and vector databases are absent.
4. Add preservation tests for the existing HTML control IDs and static assets.
5. Add safety assertions that server/user filenames and message content are not interpolated into `innerHTML`. The current filename rendering is expected to fail until Task 8.
6. Lock and sync the environment.

### Verification

```bash
uv lock
uv sync --group dev
uv run pytest tests/test_ui_assets.py -v
```

Record the known frontend safety failure; do not redesign the UI in this task.

### Completion condition

The test runner works, production dependencies are explicit, and the lockfile contains no `llama-cpp-python`.

---

## Task 2: Add clean configuration, DTOs, and atomic persistence

### Files

- Create `src/config.py`
- Create `src/models.py`
- Create `tests/test_models.py`

Do not delete `src/core` yet.

### Interfaces

`Settings` exposes:

- a single configurable project/data root;
- computed upload, staging, corpus, and history paths;
- LLM/embed/rerank URLs;
- HTTP timeouts;
- upload/context/batch/candidate limits;
- Docling termination grace time;
- `ensure_dirs()`.

Derived paths must be properties or be computed in `__post_init__`, so replacing `data_dir` in tests also changes every dependent path.

Public DTOs:

- `Chunk`
- `Document`
- `Message`
- `Corpus`
- `History`

Required operations:

- strict `from_dict()` boundary validation;
- `to_dict()` round trip;
- legacy `summaries` to `documents` corpus migration;
- history filtering to user/assistant roles only;
- immutable-style `with_document()` / `without_document()` helpers;
- atomic JSON save using a sibling temporary file and `os.replace()`.

### Tests first

Cover:

1. Corpus round trip with refs and Unicode text.
2. Legacy `summaries` migration.
3. Clean history migration from legacy system/RAG messages.
4. Invalid/malformed JSON behavior with a clear exception or documented empty-file rule.
5. Atomic replace: a serialization/write failure leaves the previous file intact.
6. Settings rooted at `tmp_path` produce no path outside it.
7. Duplicate document IDs or chunk/document mismatches are rejected at load boundaries.

### Verification

```bash
uv run pytest tests/test_models.py -v
```

### Completion condition

Persistence is independently tested and no new code imports legacy `src.core` modules.

---

## Task 3: Implement the shared llama.cpp HTTP client

### Files

- Create `src/llama.py`
- Create `tests/test_llama.py`

### Interface

One concrete `LlamaClient` wraps one injected `httpx.AsyncClient` and provides:

- `stream_chat(messages, tools=None, tool_choice=None)` yielding typed content/tool-call events;
- `complete_chat(messages, max_tokens, temperature)` for document overview generation;
- `embed(texts)` returning a normalized/validated 2D numeric result;
- `rerank(query, documents)` returning scores mapped to input document indices.

Use the exact endpoint shapes demonstrated in `test.txt`. Keep URL joining explicit and predictable.

### Required behavior

1. Non-2xx responses raise one bounded, user-safe model HTTP exception.
2. SSE ignores blank/comment lines, recognizes `[DONE]`, and reports malformed JSON/choice shapes.
3. Streaming tool calls accumulate fragmented IDs, function names, and argument strings by tool-call index.
4. Embeddings parse the deployed batch shape as a list of indexed items, extract each vector from `embedding[0]`, map by item `index`, then validate row count, dimension, nonzero dimension, numeric types, and finite values.
5. Reranking validates `results[*].index`, rejects duplicates/out-of-range indices, and maps scores by index rather than response order.
6. Empty embedding/reranking inputs return immediately without an HTTP request.
7. Cancelling the consumer closes the active HTTP response context but not the shared client.

### Tests first

Use `httpx.MockTransport` for:

- direct content SSE;
- fragmented tool-call SSE;
- malformed SSE and missing `[DONE]` policy;
- embedding batch returned out of input order, nested-vector parsing, and malformed dimensions;
- reranking responses deliberately returned out of input order;
- timeout/non-2xx translation;
- cancellation cleanup.

### Verification

```bash
uv run pytest tests/test_llama.py -v
```

### Completion condition

All three llama.cpp protocols are covered without importing the OpenAI SDK or `llama_cpp`.

---

## Task 4: Build the compact in-memory RAG index

### Files

- Create `src/rag.py`
- Create `tests/conftest.py`
- Create `tests/test_rag.py`

### Interface

`RagIndex` owns an ordered chunk list, BM25 index, and normalized `float32` vector matrix. It provides:

- `rebuild(corpus)` for startup;
- `prepare_add(chunks)` returning candidate index state without mutating live state;
- `prepare_remove(file_id)` returning candidate index state;
- `install(candidate_state)` as an in-memory, non-awaiting swap;
- `search(queries, file_ids, limit)` returning ordered chunks.

Candidate state may be a small private dataclass. Do not build a generic transaction or vector-store interface.

### Retrieval behavior

1. Embed all search queries in one request.
2. Restrict candidate indices before lexical/semantic selection when file IDs are supplied.
3. Obtain bounded BM25 and dot-product rankings per query.
4. Fuse with local reciprocal rank.
5. Cap reranking at 16 candidates per query.
6. Combine each chunk's best reranking score across queries.
7. Return at most six chunks.

Use a simple tokenizer appropriate to BM25S and document its Vietnamese limitations; do not add another NLP framework.

### Tests first

Cover:

- empty corpus rebuild;
- batched rebuild;
- add embeds only new chunks;
- candidate add/remove do not mutate live state before `install()`;
- file filtering happens before final selection;
- lexical and semantic rank fusion;
- multi-query union and best rerank score;
- candidate/final caps;
- zero vectors and dimension mismatch rejection;
- reranker failure leaves live index unchanged.

### Verification

```bash
uv run pytest tests/test_rag.py -v
```

### Completion condition

The index has no model lifecycle or executor code and can prepare a new state before document commit.

---

## Task 5: Implement the disposable Docling worker

### Files

- Create `src/docling_worker.py`
- Create `tests/test_docling_worker.py`

### Worker contract

Run as:

```bash
python -m src.docling_worker \
  --input <staged-path> \
  --output <chunks-json-path> \
  --file-id <opaque-id> \
  --file-name <safe-display-name>
```

The worker:

1. Validates explicit input/output arguments.
2. Imports Docling only inside the worker module/process.
3. Uses the exact `test.py` conversion configuration for PDF and DOCX.
4. Contextualizes every emitted chunk and preserves all Docling refs.
5. Writes plain DTO-compatible JSON to a temporary output and atomically replaces the requested result path.
6. Writes errors to stderr and exits nonzero; it does not write a partial successful result.
7. Drops Docling/document/chunk references in `finally`; process exit remains the guaranteed CUDA cleanup boundary.

Do not add progress IPC unless Docling already exposes a stable page callback with negligible code. Coarse status belongs to FastAPI.

### Tests

- Test CLI argument and output/error behavior without loading a real GPU model by monkeypatching a small conversion function.
- Test chunk DTO construction, refs, Unicode, and output atomicity.
- Add an opt-in integration test for one real DOCX/PDF fixture if a suitable non-sensitive fixture exists.
- Assert `src.main`, `src.documents`, and other FastAPI modules do not import Docling conversion or CUDA cleanup APIs.

### Verification

```bash
uv run pytest tests/test_docling_worker.py -v
```

### Completion condition

The worker can be launched independently and its only successful output is validated plain JSON.

---

## Task 6: Implement document staging, subprocess lifecycle, and transaction

### Files

- Create `src/documents.py`
- Create `tests/helpers/fake_docling_worker.py`
- Create `tests/test_documents.py`

### Interface

`DocumentService` receives settings, llama client, live corpus holder, and `RagIndex`. It provides:

- `ingest(upload_name, content_or_upload, request_state)`;
- `delete(file_id)`;
- `clear()`;
- `prune_missing_uploads(corpus)` for startup;
- a small private subprocess runner method that tests can replace.

Avoid a persistent job manager. The ingest coroutine owns one subprocess and awaits it directly.
It is also the sole authority that terminates, kills, and reaps that process; active request state only exposes the current handle for cancellation coordination.

### Ingest implementation order

1. Sanitize basename, validate extension/size, create request-scoped staging.
2. Copy upload bytes without constructing a path from the client name.
3. Emit/return the coarse `processing` status and spawn the worker with `asyncio.create_subprocess_exec` and no shell.
4. Store the subprocess handle in the active request state.
5. Await exit; on cancellation terminate, wait for the configured grace period, kill if necessary, and reap.
6. Validate worker exit code and chunks JSON.
7. Clear the active subprocess handle; at this point Docling VRAM has been released.
8. Generate a bounded overview through `LlamaClient`.
9. Prepare new-vector/BM25 state through `RagIndex`.
10. Build a candidate `Corpus`.
11. In one short synchronous commit section: move staged upload to final path, atomically save corpus, swap corpus/index state.
12. Clean staging in `finally`.

If final file move succeeds but corpus save fails, remove the moved file and keep live state untouched. Startup removes unreferenced uploads left by a process crash.

### Cancellation/persistence tests

Use the fake worker to cover:

1. Safe basename and accepted extensions.
2. Successful process result and final commit.
3. Worker nonzero exit and malformed JSON.
4. Cancel while worker runs: terminate, then kill after grace when the fake ignores terminate.
5. Cancel during overview.
6. Cancel during embedding/candidate preparation.
7. Overview/embed/persistence failure rollback.
8. No live corpus/index mutation before commit.
9. Cancel after commit preserves the document.
10. Delete removes only one document/file/vector subset.
11. Delete persistence failure restores its temporarily moved upload and live state.
12. Startup prunes missing uploads and orphan final files.
13. Clear persists/installs empty state and removes committed uploads.

### Verification

```bash
uv run pytest tests/test_documents.py -v
```

### Completion condition

Every pre-commit failure leaves old persistent/live state intact, and every worker cancellation reaps its child process.

---

## Task 7: Implement the one-tool LLM agent

### Files

- Create `src/chat.py`
- Create `tests/test_chat.py`

### Tool schemas

Expose only:

```text
get_document_overviews(file_ids: 1..8 existing IDs)
search_documents(queries: 1..3 strings, file_ids: 1..8 existing IDs, limit: 1..6)
```

An exact unique filename may normalize to an ID. Unknown or ambiguous strings produce structured tool errors; they never silently select all documents.

### Agent flow

1. Build a compact first-call prompt from instructions, ready-document catalog, recent clean history, optional newly committed document, and current message.
2. Start a streaming chat completion with tools and `tool_choice="auto"`.
3. If content begins, treat the response as a direct answer and stream it to the caller.
4. If tool-call deltas begin, accumulate exactly one call and do not expose them to the browser.
5. Validate the completed call.
6. Execute overview lookup or RAG search.
7. Append the assistant tool call and structured tool result only to the request-local message list.
8. Make a second streaming completion with tools disabled and a citation/no-fabrication instruction.
9. Persist the clean user/assistant turn only after a complete nonempty response.

Reject mixed content/tool protocol and a second tool call with a controlled error. Do not create a separate semantic analyzer or planner call.

### Tests first

Cover:

- greeting/direct answer uses one LLM call and no retrieval;
- upload acknowledgement direct answer;
- overview tool and final answer;
- search tool with rewritten queries and final citations;
- follow-up uses recent clean history/catalog;
- unique filename normalization;
- unknown/ambiguous file structured error followed by clarification;
- invalid name/arguments/caps;
- empty retrieval cannot produce an asserted document fact in the supplied final prompt;
- only one tool call allowed;
- raw overview/chunks/tool messages absent from saved history;
- cancelled/failed/partial generations leave history unchanged;
- document remains available if chat is cancelled after ingest commit.

### Verification

```bash
uv run pytest tests/test_chat.py -v
```

### Completion condition

Direct chat costs one model call; document chat costs at most two; history remains clean.

---

## Task 8: Wire the FastAPI lifecycle, cancellation, API, and existing UI

### Files

- Replace `src/main.py`
- Modify `src/static/script.js`
- Extend `tests/test_api.py`
- Extend `tests/test_ui_assets.py`

Keep `src/templates/index.html` and `src/static/style.css` unless a compatibility test proves a minimal change is necessary.

### FastAPI lifecycle

1. Create one shared `httpx.AsyncClient` with explicit timeout components.
2. Load/migrate corpus and history.
3. Prune missing uploads and unreferenced orphan uploads.
4. Rebuild the RAG index before readiness; fail startup for a nonempty corpus if embedding rebuild fails.
5. Close only the shared HTTP client at shutdown.

### Active request

Keep one small `ActiveRequest` value in app state with request ID, cancellation event, pipeline task, and optional subprocess.

- `/api/chat` atomically claims the single active slot or returns HTTP 409.
- The SSE generator registers its current task, sends request ID immediately, and releases the slot in `finally`.
- `/api/stop` sets cancellation, cancels the pipeline task, and relies on document/model context managers to terminate/close active work.
- A short commit section is allowed to finish before cancellation takes effect.
- Browser disconnect follows the same cleanup path; no incomplete history is saved.

### Endpoints

Preserve:

- `POST /api/chat`
- `POST /api/stop`
- `GET /api/chat-history`
- `POST /api/clear-chat`
- `GET /api/documents`
- `GET /api/documents/{file_id}/download`
- `DELETE /api/documents/{file_id}`

Delete rejects while chat is active. Clear first cancels/awaits active cleanup, then clears persistent state. Download uses the stored document metadata and safe final path.

### SSE

Use one JSON `data:` event format containing optional:

- `request_id`
- `status`
- `content`
- `done`
- `cancelled`
- `error`

Send SSE comments as heartbeats during long worker waits. Disable proxy buffering and caching.

### Frontend changes

1. Keep chat locked for the full ingest plus agent pipeline.
2. Stop the server request, abort the browser stream, and clean UI state deterministically.
3. Refresh committed documents after done/cancel/clear/delete.
4. Render filenames, attachment names, messages, and server errors using `textContent`/DOM properties only.
5. Retain current document controls, theme, responsive layout, and sidebar preference.

### API tests

Cover:

- startup empty/nonempty rebuild and failure;
- one active request and second-request 409;
- status order from worker through agent;
- heartbeat format;
- stop during fake worker and model stream;
- active state always cleared after success/error/cancel/disconnect;
- committed document survives agent cancel;
- failed ingest never appears in `/api/documents`;
- clear, delete, history, and download contracts;
- no CUDA/model cleanup on stop/delete/clear/shutdown.

### Verification

```bash
uv run pytest tests/test_api.py tests/test_ui_assets.py -v
```

### Completion condition

The existing UI works against the new API, renders untrusted values safely, and a stop request reaps active work without affecting persistent model servers.

---

## Task 9: Add live evaluation, remove legacy code, and verify the system

### Files

- Create `tests/fixtures/agent_cases.json`
- Create `tests/test_agent_eval.py`
- Delete after replacement tests pass:
  - `src/api/`
  - `src/core/`
  - `src/services/`

### Agent fixture

Add 40–60 Vietnamese cases covering:

- greetings and general chat;
- upload acknowledgement;
- upload plus overview request;
- existing-file overview/structure/main-points questions;
- specific and multi-query search;
- multiple-document comparison;
- clean-history follow-ups and citation references;
- unknown and ambiguous filenames;
- empty retrieval;
- attempts to induce destructive or unsupported tools.

The live test is opt-in with `RUN_LIVE_MODEL_TEST=1`. Measure:

- direct versus overview versus search choice;
- valid native tool protocol;
- selected document IDs;
- standalone rewritten queries for follow-ups;
- final response after a tool message;
- no document claim after an empty tool result.

Targets:

- 100% structurally valid responses after allowed filename normalization;
- at least 95% correct direct/tool choice;
- at least 90% correct document selection for follow-ups;
- zero unsupported claims after empty retrieval.

### Live reranker calibration

Add a small multilingual relevance fixture with obvious positive/negative documents. Verify that larger `relevance_score` means better topical relevance for the deployed model. A factually contradictory sentence may still be highly relevant to the query, so do not use entailment or factual correctness as the score-order oracle. Do not encode score inversion without evidence from the live endpoint.

### Legacy removal

Only after the model-free replacement suite passes:

1. Point the package entrypoint solely at the new app.
2. Remove legacy API/core/services trees.
3. Confirm there are no imports of `llama_cpp`, legacy cancellation types, model paths, CUDA cleanup, or legacy executors.
4. Preserve the existing corpus/history migration path.

### Automated verification

```bash
uv run pytest -v
rg -n "llama_cpp|LlamaEmbedding|CancellationToken|ThreadPoolExecutor|cuda\.empty_cache|cuda\.synchronize" src pyproject.toml
python -m compileall -q src tests
```

The `rg` command should return no production match except a deliberate comment/test assertion if one remains.

### Manual verification

1. Start the three llama.cpp containers from the operational commands in `test.txt`.
2. Start FastAPI with one Uvicorn worker.
3. Upload a DOCX with acknowledgement and with a summary request.
4. Upload a multi-page PDF and confirm UI/API responsiveness and SSE heartbeats.
5. Observe that the Docling child exits and VRAM drops before overview/embedding starts.
6. Ask direct, overview, specific, comparison, and follow-up questions.
7. Stop during Docling, overview, embedding, and answer streaming.
8. Confirm pre-commit cancel leaves no document and post-commit cancel preserves it.
9. Download/delete a document; clear chat; restart FastAPI and confirm rebuild.
10. Run the live agent fixture and reranker calibration.

### Completion condition

The model-free suite passes, live acceptance targets pass, Docling cleanup is observed, legacy code is gone, and the final diff contains only files intentionally included in the refactor.

---

## Recommended execution order

Execute Tasks 1–9 in order. Do not begin API replacement before persistence, HTTP protocol, RAG candidate state, Docling process cleanup, and agent behavior have focused tests.

At each task:

1. Write the focused failing test.
2. Implement the smallest concrete behavior that satisfies it.
3. Run the focused test and all previously completed tests.
4. Review the diff for accidental user-file changes.
5. Commit only that task when the user requests commits.

## Final guardrails

- Do not introduce multi-user or multi-request behavior.
- Do not add a Docling daemon, queue, or job database.
- Do not let the LLM poll processing status.
- Do not let the LLM mutate files or chat state.
- Do not persist partial answers or request-local RAG/tool data.
- Do not interrupt the atomic commit section.
- Do not keep Docling alive after chunk JSON is complete.
- Prefer a few readable concrete functions over reusable abstractions that have only one caller.
