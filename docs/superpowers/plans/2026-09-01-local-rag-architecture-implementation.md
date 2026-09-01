# Local RAG Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the global-lock JSON chatbot with a durable, session-aware, asynchronous local RAG application whose agent loop is owned by OpenAI Agents SDK.

**Architecture:** Keep one FastAPI process and the three existing llama.cpp services. Store application state, SDK sessions, chunks, embeddings, and document jobs in one WAL SQLite database; publish immutable in-memory RAG snapshots; run one durable document worker; and allow up to four independent SDK runs while defending against duplicate runs within one session.

**Tech Stack:** Python 3.12, FastAPI, stdlib SQLite, OpenAI Agents SDK 0.22.0, OpenAI Python SDK 3.6.0, httpx, NumPy, BM25S, LiteParse, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-local-rag-architecture-design.md`

## Global Constraints

- Run exactly one ASGI worker; in-process snapshots are not coordinated across Uvicorn workers.
- Pin `openai-agents==0.22.0`; use `SQLiteSession`, `SessionSettings(limit=48)`, `session_input_callback`, and `Runner.run_streamed`.
- Use SDK sessions only: never combine them with manual history replay, `previous_response_id`, or `conversation_id`.
- Keep one immutable `Agent`; expose only `search_documents` and `get_document_overviews` as read-only tools.
- Keep document chunking unchanged: BGE-M3 tokenizer, at most 1,024 tokens, Markdown-aware splitting, and zero overlap.
- Keep overview generation unchanged: the first 48,000 joined characters, at most 300 Vietnamese words, and 768 output tokens.
- Bound each serialized tool result to 48,000 characters and fail closed instead of returning a truncated JSON value.
- Default gates are four LLM runs, one parser, one embedding request, one reranking request, and two RAG CPU threads.
- Upload is limited to 25 MiB, parsing to 200 pages and 300 seconds, chat input to 12,000 characters, and stored/displayed errors to 500 characters.
- Conversation model input reads at most 48 durable SDK items and retains at most 12 complete visible messages totaling at most 12,000 characters.
- Use explicit SQL and small transaction helpers. Do not add an ORM, repository forwarding layer, broker, vector database, multi-agent handoff, or WebSocket.
- Follow test-first RED-GREEN-REFACTOR for every production behavior. Opt-in live tests must not be required by the normal suite.

---

## Target File Structure

- `src/config.py`: paths, model endpoints, exact budgets, and concurrency settings.
- `src/database.py`: WAL/foreign-key initialization, numbered schema migrations, and short off-event-loop read/write transactions.
- `src/models.py`: small validated domain records and legacy JSON readers used only for migration.
- `src/model_clients.py`: llama.cpp Responses model configuration plus validated overview, embedding, and reranking calls.
- `src/parse_worker.py`: existing disposable LiteParse/Markdown chunking CLI, unchanged in behavior.
- `src/parser.py`: parser subprocess lifecycle, timeout, cancellation, and semantic-output validation.
- `src/rag.py`: immutable snapshots, CPU-offloaded construction/ranking, hybrid retrieval, and atomic snapshot capture/publication.
- `src/documents.py`: upload storage, document queries, retry/delete scheduling, download resolution, and reconciliation.
- `src/jobs.py`: durable job claiming, retry/recovery, ingest/reindex/delete execution, and snapshot publication.
- `src/sessions.py`: session metadata, SDK session factory, complete-message projection, bounded context callback, and transactional run session.
- `src/agent.py`: immutable SDK agent, dynamic snapshot instructions, two tools, run streaming, and per-session cancellation.
- `src/main.py`: dependency composition, lifespan, HTTP/SSE translation, and no domain orchestration.
- `src/templates/index.html`, `src/static/script.js`, `src/static/style.css`: session-aware chat and independent document manager.
- `tests/`: unit, integration, concurrency, migration, UI, and opt-in live evaluation coverage matching the modules above.

### Task 1: Pin the SDK and establish the SQLite/domain foundation

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/config.py`
- Create: `src/database.py`
- Modify: `src/models.py`
- Create: `tests/test_database.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Produces `Database(path: Path, busy_timeout_ms: int)` with `initialize()`, `read(fn)`, and `write(fn)`; callbacks receive a configured `sqlite3.Connection` and writes run in `BEGIN IMMEDIATE` transactions.
- Produces `DocumentRecord`, `StoredChunk`, `JobRecord`, and `SessionRecord` immutable dataclasses.
- Produces settings properties `database_path`, `uploads_dir`, `staging_dir`, `legacy_corpus_path`, and `legacy_history_path`.
- Existing JSON `Corpus`/`History` readers remain temporarily for Task 5 and are removed in Task 10.

- [ ] **Step 1: Write failing schema and rollback tests**

```python
@pytest.mark.asyncio
async def test_initialize_creates_wal_schema(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()
    names = await db.read(
        lambda conn: {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    )
    assert {"schema_meta", "sessions", "documents", "chunks", "document_jobs"} <= names
    assert await db.read(lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"


@pytest.mark.asyncio
async def test_write_rolls_back_the_entire_callback(tmp_path: Path) -> None:
    db = Database(tmp_path / "app.sqlite3", busy_timeout_ms=2_000)
    await db.initialize()

    def broken(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO sessions(id,title,created_at,updated_at) VALUES(?,?,?,?)",
            ("s1", "one", 1.0, 1.0),
        )
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        await db.write(broken)
    assert await db.read(lambda conn: conn.execute("SELECT count(*) FROM sessions").fetchone()[0]) == 0
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/test_database.py tests/test_models.py -q`

Expected: collection fails because `src.database` and the new record types do not exist.

- [ ] **Step 3: Add exact dependencies, settings, records, and schema**

Add `openai-agents==0.22.0` to project dependencies and regenerate the lock with `uv lock`. Add settings for the global constraints, including `agent_model="local"`, `agent_max_turns=4`, `session_raw_item_limit=48`, `session_visible_message_limit=12`, `session_context_chars=12_000`, `session_title_chars=80`, `job_max_attempts=3`, `job_retry_base_seconds=1.0`, `database_busy_timeout_ms=5_000`, and all five concurrency values.

Implement schema version 1 with explicit checks and indexes:

```sql
CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sessions(
  id TEXT PRIMARY KEY, title TEXT NOT NULL,
  created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE documents(
  id TEXT PRIMARY KEY, file_name TEXT NOT NULL, media_type TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('processing','ready','failed','deleting')),
  overview TEXT NOT NULL DEFAULT '', chunk_count INTEGER NOT NULL DEFAULT 0,
  error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE chunks(
  document_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, refs_json TEXT NOT NULL,
  text TEXT NOT NULL, embedding BLOB, embedding_dim INTEGER,
  PRIMARY KEY(document_id, chunk_id),
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE document_jobs(
  id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN ('ingest','reindex','delete')),
  state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed','cancelled')),
  attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,
  error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL,
  started_at REAL, finished_at REAL
);
CREATE INDEX document_jobs_claim
  ON document_jobs(state, next_attempt_at, created_at);
```

`Database.write` must serialize application writes with one `asyncio.Lock`, run the full callback via `asyncio.to_thread`, commit on success, and roll back on every `BaseException`. Every connection enables `foreign_keys`, WAL, and the bounded busy timeout.

- [ ] **Step 4: Run foundation tests and the existing suite**

Run: `uv run pytest tests/test_database.py tests/test_models.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: existing behavior remains green because new records are additive.

- [ ] **Step 5: Commit the foundation**

```bash
git add pyproject.toml uv.lock src/config.py src/database.py src/models.py tests/test_database.py tests/test_models.py
git commit -m "feat: add durable sqlite foundation"
```

### Task 2: Add validated model clients and immutable RAG snapshots

**Files:**
- Create: `src/model_clients.py`
- Modify: `src/rag.py`
- Create: `tests/test_model_clients.py`
- Modify: `tests/test_rag.py`

**Interfaces:**
- Produces `LocalModelClients.complete_overview(file_name, chunks) -> str`, `embed(texts) -> list[list[float]]`, and `rerank(query, documents) -> list[float]`.
- Produces `build_agent_model(settings, http) -> OpenAIResponsesModel` using `AsyncOpenAI(base_url=f"{llm_url}/v1", api_key="local", http_client=http)`.
- Produces `IndexSnapshot(documents, chunks, vectors, lexical)`, async `SnapshotStore.capture()`, `SnapshotStore.publication_lock`, `SnapshotStore.install_locked(candidate)`, and `RagService.build(...)`/`search(snapshot, ...)`.
- Embedding and reranking semaphores are held for one HTTP batch only; snapshot CPU work uses the supplied two-worker executor.

- [ ] **Step 1: Write failing snapshot and persistence-boundary tests**

```python
@pytest.mark.asyncio
async def test_search_keeps_the_snapshot_it_started_with() -> None:
    old = snapshot_with("old", "alpha fact", [1.0, 0.0])
    new = snapshot_with("new", "beta fact", [0.0, 1.0])
    store = SnapshotStore(old)
    models = PausingModels(query_vector=[1.0, 0.0])
    rag = RagService(models, cpu_executor=ThreadPoolExecutor(max_workers=2), **LIMITS)
    task = asyncio.create_task(rag.search(await store.capture(), ["alpha"], ["old"], 5))
    await models.embedding_started.wait()
    async with store.publication_lock:
        store.install_locked(new)
    models.release_embedding.set()
    assert [chunk.document_id for chunk in await task] == ["old"]


@pytest.mark.asyncio
async def test_build_from_persisted_vectors_never_embeds() -> None:
    models = FailingIfEmbeddedModels()
    rag = RagService(models, cpu_executor=ThreadPoolExecutor(max_workers=2), **LIMITS)
    snapshot = await rag.build(DOCUMENTS, CHUNKS, np.array([[3.0, 4.0]], dtype=np.float32))
    assert snapshot.vectors.tolist() == [[0.6, 0.8]]
```

Add protocol tests that reject wrong embedding rows/dimensions, NaN/Inf/zero vectors, wrong reranker result counts, nonfinite reranker scores, and empty overview output.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_model_clients.py tests/test_rag.py -q`

Expected: new classes and module imports fail.

- [ ] **Step 3: Implement the model and snapshot boundaries**

`IndexSnapshot` is frozen, holds tuples, and marks its normalized float32 matrix non-writeable. `SnapshotStore.capture` acquires and immediately releases the publication lock so it cannot observe the database-commit/install interval. `install_locked` is called only while that same lock is held. `RagService.build` performs NumPy normalization and BM25 construction inside `loop.run_in_executor`; it never calls the embedding model. `RagService.search` takes an explicit snapshot argument, embeds clean rewrites, performs BM25/cosine/RRF CPU ranking in the executor, reranks bounded candidates, and returns at most `min(limit, final_chunk_limit, 6)` chunks.

The overview client preserves current behavior exactly:

```python
context = "\n\n---\n\n".join(
    f"[{', '.join(chunk.refs)}]\n{chunk.text}" for chunk in chunks
)[: settings.max_context_chars]
```

Keep `LlamaClient` and `RagIndex` temporarily so old routes remain green; the new runtime stops constructing them in Task 8 and Task 10 removes them.

- [ ] **Step 4: Verify model and RAG behavior**

Run: `uv run pytest tests/test_model_clients.py tests/test_rag.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit model and snapshot support**

```bash
git add src/model_clients.py src/rag.py tests/test_model_clients.py tests/test_rag.py
git commit -m "feat: add immutable rag snapshots"
```

### Task 3: Separate document storage from the parser subprocess

**Files:**
- Create: `src/parser.py`
- Modify: `src/documents.py`
- Create: `tests/test_parser.py`
- Modify: `tests/test_documents.py`

**Interfaces:**
- Produces `ParserService.parse(document_id, file_name, source_path) -> list[Chunk]` and `ParserService.cancel_active()`.
- Produces `DocumentService.create_upload(upload) -> DocumentRecord`, `list()`, `get(id)`, `retry(id)`, `schedule_delete(id)`, `download_path(id)`, and `reconcile_files()`.
- `DocumentService.source_path(id)` is exactly `uploads_dir / id`; parser staging restores the validated original suffix for LiteParse.
- Upload returns after its document row and queued ingest job commit; it never invokes parser/model/RAG code.

- [ ] **Step 1: Write failing independent-upload and semantic-validation tests**

```python
@pytest.mark.asyncio
async def test_upload_commits_processing_job_without_parsing(harness) -> None:
    upload = UploadStub("report.pdf", b"pdf bytes", "application/pdf")
    document = await harness.documents.create_upload(upload)
    assert document.status == "processing"
    assert harness.documents.source_path(document.id).read_bytes() == b"pdf bytes"
    assert await harness.job_state(document.id) == ("ingest", "queued")
    assert harness.parser_calls == []


def test_semantic_validation_rejects_markup_only_chunks() -> None:
    with pytest.raises(DataValidationError, match="meaningful content"):
        validate_parsed_chunks([Chunk("d", "x.svg", 0, ["p. 1"], "<svg><path/></svg>")])
```

Also cover 25 MiB streaming overflow cleanup, unsafe/unsupported filename, database rollback removing the newly committed file, download denial during deletion, retry only from failed, delete superseding queued ingest/reindex work, missing files, and orphan cleanup.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_parser.py tests/test_documents.py -q`

Expected: the new service contracts are absent.

- [ ] **Step 3: Implement storage and parser lifecycle**

Stream upload reads in 1 MiB blocks into a request-scoped staging file while counting bytes. Validate and normalize the display name with the existing allowlist, atomically rename staging to `uploads/<document_id>`, then insert document plus job in one `Database.write`. If the database write fails, unlink only that explicit committed path.

Move existing process-group code into `ParserService`. Under the one-parser semaphore, create a temporary directory, hard-link or copy the committed source to `input<suffix>`, run `python -m src.parse_worker`, enforce the existing timeout and SIGTERM/SIGKILL grace, validate sequential chunk IDs/identity, then apply semantic validation: strip Markdown/XML/HTML/SVG markup and require at least eight Unicode letters or digits across all chunk text.

- [ ] **Step 4: Verify document and parser behavior**

Run: `uv run pytest tests/test_parser.py tests/test_documents.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit independent document intake**

```bash
git add src/parser.py src/documents.py tests/test_parser.py tests/test_documents.py
git commit -m "feat: separate document intake and parsing"
```

### Task 4: Implement the durable document worker and atomic publication

**Files:**
- Create: `src/jobs.py`
- Create: `tests/test_jobs.py`
- Modify: `src/documents.py`
- Modify: `src/rag.py`

**Interfaces:**
- Produces `DocumentWorker.start()`, `wake()`, `stop()`, and `recover()`.
- Job claiming is `queued -> running` in one write transaction ordered by `(next_attempt_at, created_at)`.
- Ingest: parse, validate, overview, embed new chunks, stage rows, build candidate, then publish under `SnapshotStore.publication_lock`.
- Reindex: reuse stored text/overview, embed only that document's chunks, and publish.
- Delete: build without the document, commit row deletion/job success and publish under the same gate, then unlink source.

- [ ] **Step 1: Write failing recovery, retry, and publication tests**

```python
@pytest.mark.asyncio
async def test_recover_requeues_running_jobs(tmp_runtime) -> None:
    await tmp_runtime.insert_job(state="running", attempts=1)
    await tmp_runtime.worker.recover()
    assert await tmp_runtime.states() == ["queued"]


@pytest.mark.asyncio
async def test_failed_ingest_preserves_live_snapshot(tmp_runtime) -> None:
    before = await tmp_runtime.snapshots.capture()
    tmp_runtime.models.embed_error = ValueError("invalid document embedding shape")
    await tmp_runtime.worker.run_one()
    assert await tmp_runtime.snapshots.capture() is before
    assert await tmp_runtime.document_status() == "failed"


@pytest.mark.asyncio
async def test_delete_requested_during_ingest_cannot_republish_document(tmp_runtime) -> None:
    tmp_runtime.parser.pause_after_parse = True
    running = asyncio.create_task(tmp_runtime.worker.run_one())
    await tmp_runtime.parser.paused.wait()
    await tmp_runtime.documents.schedule_delete(tmp_runtime.document_id)
    tmp_runtime.parser.resume.set()
    await running
    assert tmp_runtime.document_id not in (await tmp_runtime.snapshots.capture()).document_ids
```

Cover three-attempt retry only for `TimeoutError`, `httpx.TimeoutException`, `httpx.ConnectError`, and explicit transient 408/429/5xx model errors; deterministic validation and dimension errors fail immediately. Verify batch embedding releases its semaphore between batches by allowing a waiting query embedding to run.

- [ ] **Step 2: Run job tests and verify RED**

Run: `uv run pytest tests/test_jobs.py -q`

Expected: `src.jobs` is missing.

- [ ] **Step 3: Implement worker state transitions and publication**

Use one long-lived task and one `asyncio.Event`; the database is always the source of truth. The loop clears the event, claims one eligible job, runs it, and when no job is eligible waits either for `wake()` or the next `next_attempt_at`. On shutdown it stops claiming, cancels/settles the current operation, and leaves a running row for `recover()`.

Normalize embeddings before writing each row as contiguous float32 bytes. Reconstruct candidates from ready documents plus the currently staged processing document. Under `publication_lock`, recheck both document and job states, execute the final database transaction, then assign the already-built candidate before releasing the lock. If the state changed to `deleting`, cancel the ingest/reindex job without publishing.

Retry delay is `job_retry_base_seconds * 2 ** (attempts - 1)`. Persist only `sanitize_error(exc, 500)`; log identifiers and stack traces but never prompts/chunks.

- [ ] **Step 4: Verify worker behavior and invariants**

Run: `uv run pytest tests/test_jobs.py tests/test_documents.py tests/test_rag.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit durable jobs**

```bash
git add src/jobs.py src/documents.py src/rag.py tests/test_jobs.py tests/test_documents.py tests/test_rag.py
git commit -m "feat: add durable document worker"
```

### Task 5: Migrate legacy corpus/history without destructive cleanup

**Files:**
- Create: `src/migration.py`
- Create: `tests/test_migration.py`
- Modify: `src/database.py`
- Modify: `src/models.py`

**Interfaces:**
- Produces `migrate_legacy(settings, database, session_factory) -> MigrationReport`.
- Imports only when `legacy_import_v1` is absent; separate corpus/history markers make crash recovery idempotent.
- Never modifies the two JSON files. Copy legacy sources to `uploads/<id>` while preserving existing files.
- Valid legacy chunks enter SQLite with null embeddings; documents are `processing` with queued `reindex` jobs.

- [ ] **Step 1: Write failing absence, partial, malformed, and repeat tests**

```python
@pytest.mark.asyncio
async def test_legacy_migration_is_idempotent(migration_harness) -> None:
    migration_harness.write_corpus_one_document()
    migration_harness.write_history_two_turns()
    first = await migration_harness.run()
    second = await migration_harness.run()
    assert first.imported_documents == 1
    assert second.imported_documents == 0
    assert await migration_harness.count("documents") == 1
    assert len(await migration_harness.sdk_items()) == 4
    assert migration_harness.corpus_json.exists()
    assert migration_harness.history_json.exists()


@pytest.mark.asyncio
async def test_malformed_record_does_not_discard_valid_record(migration_harness) -> None:
    migration_harness.write_mixed_valid_and_invalid_corpus()
    report = await migration_harness.run()
    assert report.imported_documents == 1
    assert report.errors
    assert await migration_harness.count("documents") == 1
```

- [ ] **Step 2: Run migration tests and verify RED**

Run: `uv run pytest tests/test_migration.py -q`

Expected: `src.migration` is missing.

- [ ] **Step 3: Implement idempotent import**

Parse each legacy record independently so one malformed record is reported and skipped. Preserve IDs, display names, overviews, ordered chunks, and page references. A missing legacy source becomes a `failed` document with a bounded explicit error and no reindex job. Use a stable `legacy-default` SDK session only when visible legacy history exists; before adding items, read it through the Session interface and add only when empty, then write the history marker.

For embedding-signature startup handling, store `embedding_signature` in `schema_meta`. A mismatch marks ready documents `processing`, clears their embeddings, queues one reindex job per document, stores the new signature, and starts from a snapshot containing only still-compatible ready documents.

- [ ] **Step 4: Verify migration and unchanged backups**

Run: `uv run pytest tests/test_migration.py tests/test_database.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit legacy migration**

```bash
git add src/migration.py src/database.py src/models.py tests/test_migration.py
git commit -m "feat: migrate legacy rag state"
```

### Task 6: Add SDK session metadata, bounded context, and transactional turns

**Files:**
- Create: `src/sessions.py`
- Create: `tests/test_sessions.py`

**Interfaces:**
- Produces `SessionService.create()`, `list()`, `rename()`, `messages()`, `delete()`, `touch_from_first_message()`, and `sdk_session(id)`.
- `sdk_session` returns `SQLiteSession(id, database_path, session_settings=SessionSettings(limit=48))`.
- Produces pure `bounded_session_input(history_items, new_items, *, max_messages, max_chars)`.
- Produces `TransactionalSession(delegate)` whose `add_items`/`pop_item` mutate a private buffer until `commit()`; `discard()` leaves durable SDK state untouched.

- [ ] **Step 1: Write failing context and transaction tests**

```python
def test_context_keeps_only_complete_recent_visible_turns() -> None:
    history = [
        user("old " * 100), assistant("old answer"), reasoning("secret"),
        function_call("search_documents"), function_output("large stale chunk"),
        user("recent question"), assistant("recent answer"),
    ]
    result = bounded_session_input(
        history, [user("new question")], max_messages=4, max_chars=60
    )
    assert result == [user("recent question"), assistant("recent answer"), user("new question")]


@pytest.mark.asyncio
async def test_transactional_session_discards_failed_turn() -> None:
    durable = MemorySession([user("existing"), assistant("answer")])
    session = TransactionalSession(durable)
    await session.add_items([user("unfinished"), assistant("partial")])
    await session.discard()
    assert await durable.get_items() == [user("existing"), assistant("answer")]
```

Also verify item and character boundaries never split a user/assistant turn, tool items in `new_items` are retained, reasoning/tool history is excluded, session IDs never cross, title derives from the first message at 80 characters, and messages project only complete visible user/assistant pairs.

- [ ] **Step 2: Run session tests and verify RED**

Run: `uv run pytest tests/test_sessions.py -q`

Expected: `src.sessions` is missing.

- [ ] **Step 3: Implement session behavior against SDK 0.22.0**

Import `SQLiteSession` from `agents` and `SessionSettings`/`SessionInputCallback` from `agents.memory`. The callback receives `(history_items, new_items)`. Filter only prior visible message items, group them from user through assistant, discard incomplete historical turns, select whole turns from newest backward under both budgets, restore chronological order, then append all current-run items unchanged.

`TransactionalSession.get_items` delegates; `add_items` appends deep-copied items to its buffer; `pop_item` pops the buffer before delegating; `commit` performs one delegate `add_items(buffer)` and clears it; `discard` only clears it. This wrapper is passed to the runner so cancellation, model failure, tool failure, and disconnect cannot leave partial durable turns.

- [ ] **Step 4: Verify session isolation and rollback**

Run: `uv run pytest tests/test_sessions.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit session support**

```bash
git add src/sessions.py tests/test_sessions.py
git commit -m "feat: add bounded agent sessions"
```

### Task 7: Replace the custom chat loop with one OpenAI Agents SDK agent

**Files:**
- Create: `src/agent.py`
- Create: `tests/test_agent.py`
- Modify: `tests/fixtures/agent_cases.json`
- Modify: `tests/test_agent_eval.py`

**Interfaces:**
- Produces `AgentContext(snapshot: IndexSnapshot, rag: RagService)`.
- Produces immutable shared `Agent[AgentContext]` with dynamic instructions and two `@function_tool` tools.
- Produces `AgentService.stream(session_id, message) -> AsyncIterator[AgentEvent]`, `stop(session_id)`, and `stop_all()`.
- Stable events are dataclasses/literals for `start`, `status`, `delta`, `done`, `error`, and `cancelled`.

- [ ] **Step 1: Write failing tool, snapshot, stream, and rollback tests**

```python
@pytest.mark.asyncio
async def test_tools_read_the_run_snapshot_even_after_publication(agent_harness) -> None:
    old = agent_harness.snapshot(document_id="old")
    context = AgentContext(old, agent_harness.rag)
    async with agent_harness.snapshots.publication_lock:
        agent_harness.snapshots.install_locked(agent_harness.snapshot(document_id="new"))
    value = json.loads(await get_document_overviews.on_invoke_tool(
        RunContextWrapper(context=context), json.dumps({"file_ids": ["old"]})
    ))
    assert value["documents"][0]["file_id"] == "old"


@pytest.mark.asyncio
async def test_cancelled_sdk_stream_discards_the_turn(agent_harness) -> None:
    stream = agent_harness.service.stream("s1", "question")
    assert (await anext(stream)).type == "start"
    await agent_harness.service.stop("s1")
    assert await agent_harness.messages("s1") == []
```

Verify schemas enforce 1-3 queries, 1-8 file IDs, and limit 1-6; results reject unknown/not-ready IDs, carry canonical IDs/names/refs/text, and are character-bounded without cutting serialized JSON. Verify non-empty final answer, max four SDK turns, tracing disabled, exactly one active run per session, and four different sessions enter the runner concurrently.

- [ ] **Step 2: Run agent tests and verify RED**

Run: `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`

Expected: `src.agent` is missing.

- [ ] **Step 3: Implement the SDK agent and stream translation**

Use Pydantic `Annotated` list/number constraints in function signatures so the SDK generates strict schemas. Tool functions serialize compact JSON and never read `SnapshotStore`; they use `RunContextWrapper.context.snapshot` only. Dynamic instructions list ready documents from that same snapshot and preserve the current grounding/citation rules.

Construct once:

```python
document_agent = Agent[AgentContext](
    name="Local document agent",
    instructions=dynamic_instructions,
    model=responses_model,
    model_settings=ModelSettings(temperature=0.1, parallel_tool_calls=False),
    tools=[search_documents, get_document_overviews],
)
```

For each run, capture the snapshot, create `TransactionalSession`, and call `Runner.run_streamed(..., context=context, session=transaction, max_turns=4, run_config=RunConfig(tracing_disabled=True, session_settings=SessionSettings(limit=48), session_input_callback=callback))` under the four-run semaphore. Translate `ResponseTextDeltaEvent` to delta, SDK tool-called events to bounded Vietnamese status, and emit done only after `stream_events()` ends and the non-empty `final_output` is validated. Commit the transactional session only then; every exception/cancellation discards it.

- [ ] **Step 4: Verify SDK ownership and concurrency**

Run: `uv run pytest tests/test_agent.py tests/test_agent_eval.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit the SDK agent**

```bash
git add src/agent.py tests/test_agent.py tests/test_agent_eval.py tests/fixtures/agent_cases.json
git commit -m "feat: migrate chat to agents sdk"
```

### Task 8: Cut FastAPI over to session/document APIs and durable lifespan

**Files:**
- Rewrite: `src/main.py`
- Rewrite: `tests/test_api.py`
- Create: `tests/test_concurrency.py`

**Interfaces:**
- Exposes the exact session/document endpoints in the approved spec.
- `POST /api/documents` accepts only multipart upload and returns HTTP 202 before parsing.
- `POST /api/sessions/{id}/chat` accepts JSON `{"message": "..."}` and emits named application SSE data objects.
- Application lifespan initializes DB/migration/snapshot/recovery before readiness, starts one worker, and shuts down streams/worker/parser/clients/executor in that order.

- [ ] **Step 1: Replace API tests with failing contracts**

```python
def test_upload_returns_202_while_parser_is_blocked(app_harness) -> None:
    app_harness.parser.block()
    with TestClient(app_harness.app) as client:
        response = client.post(
            "/api/documents", files={"file": ("report.pdf", b"pdf", "application/pdf")}
        )
        assert response.status_code == 202
        assert response.json()["status"] == "processing"
        assert client.post("/api/sessions").status_code == 201


@pytest.mark.asyncio
async def test_four_sessions_run_without_global_conflict(async_client, runner_probe) -> None:
    session_ids = [(await async_client.post("/api/sessions")).json()["id"] for _ in range(4)]
    responses = await asyncio.gather(*[
        async_client.post(f"/api/sessions/{sid}/chat", json={"message": "hello"})
        for sid in session_ids
    ])
    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert runner_probe.max_concurrency == 4
```

Cover session CRUD/messages/stop, per-session duplicate 409, document list/status/retry/delete/download, pre-stream 4xx validation, post-stream error/cancelled events, disconnect rollback, deletion during ingestion, restart recovery, and shutdown with an active parser/stream.

- [ ] **Step 2: Run API/concurrency tests and verify RED**

Run: `uv run pytest tests/test_api.py tests/test_concurrency.py -q`

Expected: old combined `/api/chat` contracts fail the new endpoint assertions.

- [ ] **Step 3: Implement a composition-only runtime**

Create dependencies in lifespan: configured `httpx.AsyncClient`, explicit Responses model, `Database`, executor, semaphores, `LocalModelClients`, `RagService`, initial persisted snapshot, `SnapshotStore`, `ParserService`, `DocumentService`, `SessionService`, `DocumentWorker`, and `AgentService`. Run migration and job recovery before assigning `app.state.runtime`; then start the worker.

Routes validate inputs and call one service method. `FileResponse` receives the server-derived source path, original validated filename, and stored media type so `Content-Disposition` is safely encoded without accepting a client path. SSE encodes objects as `data: <compact-json>\n\n`, sends heartbeat comments without changing application state, and cancels the exact session run on disconnect. Session deletion first stops and settles that session, clears SDK items through `SQLiteSession.clear_session`, closes the SDK session, then deletes metadata. No route owns parser/model/RAG transaction logic.

- [ ] **Step 4: Verify HTTP, recovery, concurrency, and shutdown**

Run: `uv run pytest tests/test_api.py tests/test_concurrency.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit the backend cutover**

```bash
git add src/main.py tests/test_api.py tests/test_concurrency.py
git commit -m "feat: expose async session document api"
```

### Task 9: Make the SPA session-aware and documents independent from chat

**Files:**
- Modify: `src/templates/index.html`
- Create: `src/static/state.mjs`
- Rewrite: `src/static/script.js`
- Modify: `src/static/style.css`
- Rewrite: `tests/test_ui_assets.py`

**Interfaces:**
- UI owns `selectedSessionId`, a map of active stream controllers by session ID, and a document polling timer.
- Upload is an immediate document action and never enters chat `FormData`.
- Switching sessions refreshes persisted messages and never aborts another session's stream.
- Poll documents every 1.5 seconds only while a status is `processing` or `deleting`.

- [ ] **Step 1: Write failing DOM/API behavior tests**

Use the existing HTML parser for accessible controls and execute the pure ES module with the installed Node runtime for state transitions:

```python
def test_template_separates_upload_from_prompt_form() -> None:
    tree = parse_template()
    assert tree.has_button("new-session-btn")
    assert tree.has_list("sessions-list")
    assert tree.has_input("document-file-input")
    assert not tree.is_descendant("document-file-input", "prompt-form")


def test_document_polling_only_runs_for_nonterminal_states() -> None:
    assert run_state_function("shouldPollDocuments", [{"status": "ready"}]) is False
    assert run_state_function("shouldPollDocuments", [{"status": "failed"}]) is False
    assert run_state_function("shouldPollDocuments", [{"status": "processing"}]) is True
    assert run_state_function("shouldPollDocuments", [{"status": "deleting"}]) is True
```

`run_state_function` invokes `node --input-type=module -e` to import `src/static/state.mjs`, parse literal JSON input, call the named exported pure function, and parse its JSON output. Cover New chat, rename/delete selected session, message reload, independent upload, status polling start/stop, retry/delete/download visibility, stream delta routing only into its session buffer, and stop for selected session. API integration tests remain responsible for fetch/HTTP behavior rather than mocking the browser framework.

- [ ] **Step 2: Run UI tests and verify RED**

Run: `uv run pytest tests/test_ui_assets.py -q`

Expected: required session controls and independent uploader are absent.

- [ ] **Step 3: Implement the approved interaction model**

Keep the current visual language and responsive layout. Add a compact session rail with New chat and rename/delete controls; move file selection/upload into the document sidebar; render status/error/chunk count per document; and keep the prompt form text-only. Put only pure state functions (`shouldPollDocuments`, document action selection, and per-session stream reduction) in `state.mjs`; `script.js` owns DOM and fetch effects and imports those functions as an ES module.

On initialization load/create a session, then concurrently load sessions, its messages, and documents. Store each active response by session ID so a background stream can finish after selection changes; only render it when that session is selected, otherwise refresh from persisted messages when selected later. Start document polling after upload/retry/delete and stop when every returned document is ready or failed.

- [ ] **Step 4: Verify UI assets and backend contracts together**

Run: `uv run pytest tests/test_ui_assets.py tests/test_api.py -q`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Commit the SPA cutover**

```bash
git add src/templates/index.html src/static/state.mjs src/static/script.js src/static/style.css tests/test_ui_assets.py
git commit -m "feat: add sessions and independent documents ui"
```

### Task 10: Remove superseded code and complete regression verification

**Files:**
- Delete: `src/chat.py`
- Delete: `src/llama.py`
- Modify: `src/models.py`
- Modify: `src/documents.py`
- Modify: `src/rag.py`
- Rewrite: `README.md`
- Modify: all affected `tests/test_*.py`

**Interfaces:**
- Removes `History`, `LiveHistory`, custom streamed tool-call assembly, `LiveCorpus`, old transactional JSON corpus writes, `RagIndex`, the global active request slot, combined upload/chat, `/api/chat-history`, and `/api/clear-chat`.
- Keeps legacy JSON decoding isolated in `src/migration.py`; runtime modules never write JSON state.
- Documents exact one-worker startup, three model services, durable statuses/jobs, session APIs, limits, optional live tests, and recovery behavior.

- [ ] **Step 1: Add failing architecture and performance regressions**

```python
def test_runtime_imports_no_superseded_chat_loop() -> None:
    imported = imported_top_level_modules(Path("src/main.py"))
    assert "chat" not in imported
    assert "llama" not in imported


@pytest.mark.asyncio
async def test_large_snapshot_build_does_not_stall_event_loop(snapshot_factory) -> None:
    ticks = 0
    running = True
    async def ticker() -> None:
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0)
    task = asyncio.create_task(ticker())
    await snapshot_factory.build(5_000)
    running = False
    await task
    assert ticks > 10
```

Add mutation-oriented acceptance tests for every invariant in the spec: no partial ready document, no snapshot replacement on failed preparation, no re-embedding on add/delete/restart, no lost job, no partial cancelled turn, no cross-session history, shared documents, snapshot-local tool access, and no measured blocking work on the event loop.

- [ ] **Step 2: Run architecture tests and verify RED**

Run: `uv run pytest -q`

Expected: superseded modules/symbols still exist or the new regression is not yet satisfied.

- [ ] **Step 3: Delete compatibility paths and update documentation**

Remove only code made unreachable by Tasks 1-9. Move the small legacy JSON value validators needed by migration into `src/migration.py`; remove all runtime JSON save methods. Ensure every public module has one responsibility from the target structure and that `main.py` contains only construction/routes/translation.

Update README with:

```text
uv sync
docker compose up -d
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1
uv run pytest -q
RUN_LIVE_MODEL_TESTS=1 uv run pytest -m live_model -q
RUN_PARSE_INTEGRATION=1 uv run pytest -m parse_integration -q
```

Describe that `data/app.sqlite3` is authoritative and the two legacy JSON files remain recovery backups after first import.

- [ ] **Step 4: Run fresh complete verification**

Run: `uv run pytest -q`

Expected: all normal tests pass with only explicitly opt-in parse/live tests skipped.

Run: `uv run python -m compileall -q src tests`

Expected: exit 0.

Run: `git diff --check`

Expected: exit 0 with no output.

If local llama.cpp services are healthy, run: `RUN_LIVE_MODEL_TESTS=1 uv run pytest -m live_model -q`.

Expected: agent tool-choice, grounded answer/citation, one-run latency, and four-run throughput cases pass. If services are unavailable, report the skipped external verification separately and do not describe it as passing.

- [ ] **Step 5: Commit cleanup and docs**

```bash
git add -A
git commit -m "refactor: complete durable async rag architecture"
```

## Final Requirement Audit

Before branch completion, reread the approved spec line by line and record evidence for all ten consistency invariants and all acceptance criteria. Run `git status --short`, inspect the complete branch diff against its base, and use `superpowers:verification-before-completion` followed by `superpowers:finishing-a-development-branch`.
