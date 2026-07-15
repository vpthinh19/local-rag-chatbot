# Local RAG Mini-Agent Implementation Plan

## Goal

Replace the legacy backend with a small FastAPI RAG agent that calls three persistent llama.cpp HTTP servers and spawns one disposable LiteParse subprocess group per upload. Preserve the existing browser experience and JSON data while deleting the old model/service/cancellation architecture only after equivalent tests pass.

## Implementation principles

- Treat legacy code as contract/reference, not code to port class by class.
- Keep one Uvicorn worker and one active chat pipeline.
- Keep all persistent application state in FastAPI; isolate LiteParse, OCR, conversion, and Markdown chunking in one temporary worker.
- Direct LLM answer or at most one read-only tool call per user turn.
- Build candidate state before commit; cancel discards only uncommitted work.
- Add no framework, queue, database, vector store, or generic abstraction without a demonstrated need.
- Preserve all unrelated user work in the dirty worktree unless a task explicitly brings a file into scope.
- Use concise commits and the repository's configured human identity when commits are requested.

## Target structure

```text
src/
  main.py
  config.py
  models.py
  llama.py
  rag.py
  parse_worker.py
  documents.py
  chat.py
  templates/index.html
  static/style.css
  static/script.js
tests/
  fixtures/agent_cases.json
  helpers/fake_parse_worker.py
  test_models.py
  test_llama.py
  test_rag.py
  test_parse_worker.py
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
   - `fastapi[standard]`
   - `httpx`
   - `liteparse`
   - `numpy`
   - `semantic-text-splitter`
   - `tokenizers`
2. Add a dev dependency group containing `pytest` and `pytest-asyncio`.
3. Confirm `docling`, `torch`, `chonkie`, `llama-cpp-python`, `model2vec`, OpenAI SDK, agent frameworks, task queues, and vector databases are absent.
4. Verify `libreoffice --headless --version` for DOCX conversion and prefetch `BAAI/bge-m3` through `Tokenizer.from_pretrained()` once during environment setup. Do not commit cache contents.
5. Add preservation tests for the existing HTML control IDs and static assets.
6. Add safety assertions that server/user filenames and message content are not interpolated into `innerHTML`. The current filename rendering is expected to fail until Task 8.
7. Lock and sync the environment.

### Verification

```bash
uv lock
uv sync --group dev
libreoffice --headless --version
uv run python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('BAAI/bge-m3')"
uv run pytest tests/test_ui_assets.py -v
```

Record the known frontend safety failure; do not redesign the UI in this task.

### Completion condition

The test runner works, production dependencies are explicit, and the lockfile contains none of the excluded parser/model/framework packages.

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
- parse process-group termination grace time;
- maximum parse pages and the BGE-M3 tokenizer identifier or local path;
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

## Task 5: Implement the disposable LiteParse and chunking worker

### Files

- Create `src/parse_worker.py`
- Create `tests/test_parse_worker.py`

### Worker contract

Run as:

```bash
python -m src.parse_worker \
  --input <staged-path> \
  --output <chunks-json-path> \
  --file-id <opaque-id> \
  --file-name <safe-display-name>
```

The worker:

1. Validates explicit input/output arguments.
2. Imports LiteParse, `tokenizers`, and `semantic_text_splitter` only inside the worker module/process.
3. Runs LiteParse locally with selective OCR, Tesseract language `vie+eng`, 150 DPI, Markdown output, the configured maximum page count, and very-small-text preservation disabled.
4. Reads `page_num` and `page.markdown` from every parsed page, joins the document while recording each page's character span, and rejects an empty usable result.
5. Resolves the `BAAI/bge-m3` tokenizer from the configured local path/cache. Missing tokenizer data is a controlled error; never fall back silently to character counts.
6. Uses `MarkdownSplitter.from_huggingface_tokenizer(..., 1024, overlap=0)` and `chunk_indices()` over the full document, not one independent split per page.
7. Maps each chunk's start/end offsets to deterministic page refs with `bisect`, builds stable DTO-compatible chunk IDs, and validates nonempty text and the 1024-token payload bound with special tokens excluded.
8. Writes plain DTO-compatible JSON to a sibling temporary output and atomically replaces the requested result path.
9. Writes bounded errors to stderr and exits nonzero; it does not write a partial successful result.
10. Drops parser/page/Markdown/chunk references in `finally`; process-group exit remains the guaranteed native memory, OCR-thread, and converter-child cleanup boundary.

Do not add progress IPC. Coarse status and SSE heartbeats belong to FastAPI.

### Tests

- Test CLI argument and output/error behavior by monkeypatching a small parse function; the normal suite must not require LibreOffice, OCR, network access, or model weights.
- Test synthetic multi-page Markdown spans, Unicode, chunks crossing page boundaries, offset-to-page refs, stable IDs, 1024-token limit, zero overlap, and output atomicity.
- Use a small callback tokenizer in pure mapping tests and separately assert that production construction selects the configured BGE-M3 tokenizer.
- Add opt-in/local integration checks for `docs/test.pdf` OCR and `docs/DACSN.docx`; never copy their contents into committed fixtures.
- Assert `src.main`, `src.documents`, and other FastAPI modules do not import LiteParse, Tesseract, LibreOffice wrappers, `tokenizers`, or `semantic_text_splitter`.

### Verification

```bash
uv run pytest tests/test_parse_worker.py -v
```

### Completion condition

The worker can be launched independently, preserves page refs through structure-aware Markdown chunks, and its only successful output is validated plain JSON.

---

## Task 6: Implement document staging, subprocess lifecycle, and transaction

### Files

- Create `src/documents.py`
- Create `tests/helpers/fake_parse_worker.py`
- Create `tests/test_documents.py`

### Interface

`DocumentService` receives settings, llama client, live corpus holder, and `RagIndex`. It provides:

- `ingest(upload_name, content_or_upload, request_state)`;
- `delete(file_id)`;
- `clear()`;
- `prune_missing_uploads(corpus)` for startup;
- a small private subprocess runner method that tests can replace.

Avoid a persistent job manager. The ingest coroutine owns one subprocess and awaits it directly.
It is also the sole authority that terminates, kills, and reaps that process group; active request state only exposes the current handle for cancellation coordination.

### Ingest implementation order

1. Sanitize basename, validate extension/size, create request-scoped staging.
2. Copy upload bytes without constructing a path from the client name.
3. Emit/return the coarse `processing` status and spawn the worker with `asyncio.create_subprocess_exec`, no shell, and `start_new_session=True` on the Linux target.
4. Store the subprocess handle in the active request state.
5. Await exit; on cancellation signal the worker process group, wait for the configured grace period, kill the group if necessary, and reap the direct child.
6. Validate worker exit code and chunks JSON.
7. Clear the active subprocess handle; at this point LiteParse native memory, OCR threads, and converter children have been released.
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
4. Cancel while worker runs: terminate its process group, then kill after grace when the fake worker and a fake grandchild ignore termination.
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

Every pre-commit failure leaves old persistent/live state intact, and every worker cancellation reaps its direct child without leaving converter descendants.

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
- no parser, tokenizer, CUDA, or model cleanup in FastAPI on stop/delete/clear/shutdown.

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

Only after the external-model-free replacement suite passes:

1. Point the package entrypoint solely at the new app.
2. Remove legacy API/core/services trees.
3. Confirm there are no imports of `llama_cpp`, Docling/Torch, Chonkie/Model2Vec, legacy cancellation types, model paths, CUDA cleanup, or legacy executors.
4. Preserve the existing corpus/history migration path.

### Automated verification

```bash
uv run pytest -v
rg -n "docling|torch|chonkie|model2vec|llama_cpp|LlamaEmbedding|CancellationToken|ThreadPoolExecutor|cuda\.empty_cache|cuda\.synchronize" src pyproject.toml
python -m compileall -q src tests
```

The `rg` command should return no production match except a deliberate comment/test assertion if one remains.

### Manual verification

1. Start the three llama.cpp containers from the operational commands in `test.txt`.
2. Start FastAPI with one Uvicorn worker.
3. Upload `docs/DACSN.docx` with acknowledgement and with a summary request; verify LibreOffice conversion, readable Markdown, and page refs.
4. Upload the image-only `docs/test.pdf`; verify selective Vietnamese/English OCR, UI/API responsiveness, and SSE heartbeats.
5. Observe that the LiteParse process group exits and RAM is reclaimed before overview/embedding starts; model-server VRAM must remain effectively unchanged during parsing.
6. Ask direct, overview, specific, comparison, and follow-up questions.
7. Stop during LiteParse/OCR, overview, embedding, and answer streaming; confirm no LibreOffice/converter process remains.
8. Confirm pre-commit cancel leaves no document and post-commit cancel preserves it.
9. Download/delete a document; clear chat; restart FastAPI and confirm rebuild.
10. Run the live agent fixture and reranker calibration.

### Completion condition

The external-model-free suite passes, live acceptance targets pass, LiteParse/OCR quality and process-group cleanup are observed, legacy code is gone, and the final diff contains only files intentionally included in the refactor.

---

## Recommended execution order

Execute Tasks 1–9 in order. Do not begin API replacement before persistence, HTTP protocol, RAG candidate state, LiteParse process-group cleanup, and agent behavior have focused tests.

At each task:

1. Write the focused failing test.
2. Implement the smallest concrete behavior that satisfies it.
3. Run the focused test and all previously completed tests.
4. Review the diff for accidental user-file changes.
5. Commit only that task when the user requests commits.

## Final guardrails

- Do not introduce multi-user or multi-request behavior.
- Do not add a parser daemon, queue, or job database.
- Do not let the LLM poll processing status.
- Do not let the LLM mutate files or chat state.
- Do not persist partial answers or request-local RAG/tool data.
- Do not interrupt the atomic commit section.
- Do not keep LiteParse, its worker, or converter children alive after chunk JSON is complete.
- Prefer a few readable concrete functions over reusable abstractions that have only one caller.
