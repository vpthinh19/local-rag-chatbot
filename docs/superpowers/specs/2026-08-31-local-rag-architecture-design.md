# Local RAG Architecture Redesign

Status: approved in design review on 2026-08-31. Implementation is intentionally deferred.

## Purpose

Refactor the existing single-user local RAG chatbot into a small, robust modular monolith that:

- supports real asynchronous progress across chat sessions and document work;
- uses the OpenAI Agents SDK instead of a hand-written agent loop;
- persists conversations, documents, embeddings, and background jobs safely;
- keeps chat independent from upload, download, listing, indexing, and deletion;
- preserves retrieval quality and citation behavior;
- remains short enough to understand without framework-shaped abstraction layers.

The priority order is correctness, recoverability, clarity, and then throughput. Concurrency is bounded by the hardware and the measured behavior of each local model service.

## Current Baseline

The current application is a FastAPI process backed by three llama.cpp services:

- LLM on port 8080, configured with four slots;
- embedding model on port 8081;
- reranker on port 8082.

The existing test suite passes: 170 tests passed and 7 were skipped. Live agent evaluation passed its two cases, but the current API has one global request lock: eight concurrent chat requests produce one success and seven `409` responses.

Relevant measurements on the current machine:

- four concurrent Agents SDK tool loops complete successfully without the application-level `409` restriction;
- embedding and reranking throughput does not improve when identical requests are issued concurrently;
- one parser process peaks at roughly 700–750 MiB RSS;
- rebuilding BM25 for 5,000 chunks can block the event loop for roughly 267 ms if executed inline;
- JSON persistence for a 5,000-chunk corpus can block the event loop for roughly 68 ms;
- llama.cpp exposes a working Responses API and completed an end-to-end OpenAI Agents SDK tool loop in the feasibility probe.

These observations justify bounded per-resource concurrency rather than unconstrained task creation.

## Scope

This design covers one cohesive application-level refactor:

1. SQLite persistence and migration from the existing JSON state.
2. Durable document jobs and independent document APIs.
3. Immutable, copy-on-write RAG snapshots with persisted embeddings.
4. OpenAI Agents SDK chat sessions and streaming.
5. Session-aware UI and independent document management.
6. Recovery, concurrency, evaluation, and performance regression tests.

Implementation should be staged in vertical milestones, but every milestone must converge on the interfaces and invariants in this document.

## Non-goals

- Multi-user authentication, authorization, tenants, or per-user data.
- Per-session document collections; every ready document belongs to one shared corpus.
- Redis, Celery, a distributed message broker, or separate application workers.
- A vector database or an ORM.
- Multi-agent handoffs or specialist agents. One well-defined document agent is sufficient.
- Document mutation through agent tools.
- Cloud-hosted conversation state or cloud tracing by default.
- WebSockets; SSE is sufficient for chat and lightweight polling is sufficient for document status.
- Incremental BM25 maintenance before rebuilding a snapshot becomes a measured bottleneck.

## Chosen Approach

Use a modular monolith: one FastAPI process, one SQLite database in WAL mode, one durable document-job loop, parser subprocesses, and three existing llama.cpp model services.

The rejected alternatives are:

- ephemeral `asyncio.Task` ingestion, because restart loses jobs and leaves ambiguous document state;
- Redis/Celery or a separate ingestion service, because its deployment and distributed failure modes are not justified for one user;
- retaining the custom chat-completions tool loop, because it duplicates orchestration, state, streaming, and tool protocol behavior already provided by the Agents SDK.

## Architecture

```text
FastAPI application
├── Chat API ──> Agents SDK ──> read-only document tools ──> RAG snapshot
├── Session API ──> session metadata + Agents SDK SQLiteSession
├── Document API ──> files + SQLite document jobs
├── Document worker
│   ├── parser subprocess
│   ├── overview generation
│   ├── embedding client
│   └── candidate snapshot publication
└── SQLite
    ├── application tables
    └── Agents SDK session tables
```

Module responsibilities:

- `main`: construct dependencies, manage lifespan, and declare routes. It contains no document, retrieval, or agent orchestration logic.
- `database`: initialize SQLite, run small numbered SQL migrations, and expose short transaction helpers. Do not introduce repositories that only forward calls.
- `sessions`: manage session metadata and create Agents SDK `SQLiteSession` objects.
- `agent`: define one agent, its instructions, its two tools, run limits, context selection, and stream-event translation.
- `documents`: validate uploads and expose document operations and state.
- `jobs`: claim and run durable document jobs.
- `parser`: retain the existing isolated parser subprocess boundary.
- `rag`: build immutable snapshots and perform hybrid retrieval.
- `models`: retain only domain and API data types that provide real validation value.
- `model_clients`: configure the Agents SDK Responses model and provide minimal embedding/reranking clients for local endpoints not covered by the agent runtime.

The names may follow existing module names where that reduces churn. The responsibility boundaries are normative; a particular filename is not.

## Persistence

Use one application-owned SQLite database, `data/app.sqlite3`, configured with WAL, foreign keys, and a bounded busy timeout. SQL is explicit and small; schema migrations are numbered functions recorded in `schema_meta`.

Application tables:

### `sessions`

- `id` — stable opaque identifier;
- `title` — initially derived by truncating the first user message, without an extra model call;
- `created_at` and `updated_at`.

Agents SDK message items remain in the SDK-owned session tables in the same database. Application code must use the Session interface rather than editing those rows directly.

### `documents`

- `id` — stable opaque identifier and storage filename;
- `file_name` — original display/download name;
- `media_type`;
- `status` — `processing`, `ready`, `failed`, or `deleting`;
- `overview`;
- `chunk_count`;
- `error` — bounded user-displayable failure text;
- `created_at` and `updated_at`.

Do not store an arbitrary filesystem path. Derive the committed path from the validated document ID.

### `chunks`

- `document_id` and ordered `chunk_id`;
- serialized source-page references;
- text;
- normalized float32 embedding as a BLOB;
- embedding dimension.

Persist embeddings so application restart, add, and delete operations never re-embed unchanged chunks.

### `document_jobs`

- `id` and `document_id`;
- `operation` — `ingest`, `reindex`, or `delete`;
- `state` — `queued`, `running`, `succeeded`, `failed`, or `cancelled`;
- `attempts`, `next_attempt_at`, and bounded `error`;
- `created_at`, `started_at`, and `finished_at`.

The table is the durable queue. An in-memory `asyncio.Event` only wakes the worker; losing the event cannot lose work. Job payloads contain identifiers, not serialized callables or large document data.

`document_id` is a logical subject ID rather than a cascading foreign key so a completed delete-job record can survive deletion of the document row.

Store an embedding signature in schema metadata. A configured embedding-model change schedules a controlled corpus reindex, even if the new model has the same vector dimension.

## Session and Agent Design

One immutable Agent definition is shared by all runs. Each conversation has a distinct `session_id` and Agents SDK `SQLiteSession`. Use the local llama.cpp Responses endpoint through an explicit Agents SDK/OpenAI client configuration; do not rely on an SDK default model.

Use exactly one conversation-state strategy: SDK sessions. Do not combine sessions with manual history replay, `previous_response_id`, or a server-managed conversation ID.

The current custom `History`, `LiveHistory`, manual recent-history replay, fragmented tool-call assembly, and second completion pass are removed.

The agent exposes two read-only function tools:

- `search_documents(queries, file_ids, limit)` for detailed retrieval;
- `get_document_overviews(file_ids)` for summaries, outlines, and broad comparisons.

Both tools resolve only documents present in the captured ready snapshot. They return structured, size-bounded results containing canonical file IDs, display names, page references, and text. Upload, retry, delete, and download are never agent tools.

At run start, capture one RAG snapshot into the Agents SDK run context. Dynamic instructions list the ready documents from that snapshot, and both tools read the same snapshot. A corpus publication during the run therefore cannot make the prompt catalog disagree with tool resolution.

Each run has a small maximum turn count. The Agents SDK owns model/tool iteration and tool schema validation. No handoff or second agent is configured.

### Bounded conversation context

Durable history must not imply unbounded model input. Pin a validated Agents SDK version and use its supported `SessionSettings` and `session_input_callback` surfaces to select a bounded suffix of complete turns.

The callback is a small pure function, not a replacement agent loop. It:

- preserves complete recent user/assistant turns within the configured item and character budget;
- excludes old reasoning, tool-call, and tool-output items from later turns;
- leaves tool items from the current SDK run intact;
- never splits a function call from its output;
- retains the full durable session for UI/history until the user deletes it.

Default bounds match the current proven behavior: fetch at most 48 raw session items, then retain at most 12 visible user/assistant messages and 12,000 characters. Keep these values in settings and change them only against conversation evals.

This prevents stale retrieval results and large tool payloads from being replayed indefinitely while preserving normal conversational follow-up.

### Streaming and cancellation

Use `Runner.run_streamed`. Translate only stable application events to SSE:

```text
start -> status/tool status -> delta* -> done
                              \-> error or cancelled
```

Do not emit `done` until the SDK stream has fully settled. Only a successful complete turn is visible in persistent chat history. Cancellation or client disconnect cancels the run and discards its incomplete user/assistant turn. Tests must verify the exact SDK behavior; if the built-in session writes early, wrap the Session interface transactionally rather than duplicating the runner.

Tracing to OpenAI is disabled by default because the application and corpus are local. Local structured logs include the relevant session, job, and document identifiers but never complete document chunks or prompts.

## HTTP API

Session endpoints:

```text
POST   /api/sessions
GET    /api/sessions
PATCH  /api/sessions/{session_id}
GET    /api/sessions/{session_id}/messages
DELETE /api/sessions/{session_id}
POST   /api/sessions/{session_id}/chat      # SSE
POST   /api/sessions/{session_id}/stop
```

Document endpoints:

```text
POST   /api/documents                       # multipart, returns 202
GET    /api/documents
GET    /api/documents/{document_id}/download
POST   /api/documents/{document_id}/retry
DELETE /api/documents/{document_id}         # schedules durable deletion
```

The messages endpoint returns only complete user and assistant messages; reasoning, function calls, and function outputs remain internal SDK state. Deleting an active session first cancels and settles its run, then clears SDK items and session metadata.

The old combined `/api/clear-chat` behavior is removed. Session deletion and document deletion are deliberately separate operations.

Upload is never part of a chat request. `POST /api/documents` atomically establishes a committed file and a queued job, returns the document ID and `processing` status, and lets the UI continue independently.

Download is allowed while the committed source exists, except during or after deletion. The server controls the storage path and safely encodes the original filename in `Content-Disposition`.

Validation errors return `4xx` JSON before a stream starts. Once SSE starts, failures use one `error` event followed by stream closure. Internal stack traces are logged locally and never returned to the browser.

## Durable Document Work

### Upload

1. Stream the bounded upload to `data/staging` while validating size, extension, and supported media type.
2. Assign an opaque document ID and atomically rename the file into `data/uploads`.
3. In one SQLite transaction, create the `processing` document and queued `ingest` job.
4. Wake the worker and return HTTP 202.
5. If the database transaction fails, remove the just-committed file. Startup reconciliation removes a file orphaned by process death between rename and database commit.

### Job execution

The worker claims the oldest eligible job in a short transaction and changes it from `queued` to `running`. Parsing, model calls, and snapshot construction happen outside the claim transaction.

An ingest job:

1. parses in the existing subprocess boundary with timeout and termination grace;
2. validates that parsing produced at least one chunk with valid source references and at least eight Unicode letters or digits after Markdown and XML/SVG/HTML markup is removed; markup-only output is invalid;
3. generates or validates the document overview;
4. embeds only new chunks in bounded batches;
5. stores staged chunks and embeddings;
6. builds a candidate snapshot outside the event loop;
7. enters the short snapshot-publication gate, commits the document as `ready` and the job as `succeeded`, installs the candidate, and releases the gate.

Candidate construction never holds the publication gate. Search holds it only long enough to capture the current reference, then performs the complete retrieval without a lock. If the final database transaction fails, the candidate is never installed. A crash after commit but before the in-memory assignment cannot serve inconsistent requests because startup reconstructs the snapshot before accepting traffic. If a document was marked for deletion during ingestion, the worker must not publish it.

### Retry and recovery

Retry only timeouts, connection failures, and explicitly transient model-server failures. Use bounded backoff and at most three attempts. Invalid files, empty parse output, unsupported content, dimension mismatch, and deterministic validation failures fail immediately.

At startup, change stale `running` jobs back to `queued`, reconcile staging and committed files, and wake the worker. A process crash at any point therefore leaves a recoverable database state rather than an in-memory-only task.

### Delete

Deletion is also durable:

1. set the document to `deleting` and enqueue a `delete` job;
2. cancel or supersede any queued ingest/reindex job for that document;
3. build a candidate snapshot without the document;
4. enter the publication gate, transactionally delete document/chunk rows and mark the delete job complete, install the candidate, and release the gate;
5. remove the source file, treating a failed unlink as an orphan-cleanup concern rather than restoring deleted searchable data.

A search that already captured the old snapshot may finish. New searches use the new snapshot and cannot see the deleted document.

## RAG Snapshot Design

`IndexSnapshot` is immutable and internally aligned:

- ready document metadata;
- an ordered tuple of chunks;
- a read-only normalized embedding matrix;
- a BM25 index built from the same ordered chunks.

Search captures `snapshot = current_snapshot` once under the short publication gate and uses that value for the full retrieval request. Publication prepares a complete candidate before entering the gate, commits the matching database state, and replaces the current reference before releasing the gate. Readers never observe a partially built index, and long retrieval never holds a lock.

Every corpus mutation creates a new snapshot. The expensive model behavior differs:

- add: parse and embed only new chunks, then combine existing persisted embeddings with the new embeddings and rebuild in-memory BM25/vector structures;
- delete: omit the removed chunks and embeddings, then rebuild the in-memory structures without any embedding call;
- restart: load ready chunks and embeddings from SQLite and rebuild in-memory structures without any embedding call;
- embedding-signature change: explicitly re-embed the corpus through durable reindex jobs.

BM25 is rebuilt for both addition and deletion because corpus-wide document frequency and IDF values change. Incremental BM25 is deferred until snapshot construction is a measured bottleneck.

Retrieval stages remain hybrid and bounded: query rewriting by the agent, query embedding, BM25 and cosine candidates, reciprocal-rank fusion, and reranking. Citation metadata remains attached to chunks throughout the pipeline.

## Concurrency Model

There is no application-wide request lock.

Run exactly one ASGI application worker. Async tasks, the bounded thread pool, and parser subprocesses provide concurrency inside that worker. Multiple Uvicorn application workers are unsupported because each would own a different in-memory snapshot; supporting them would require cross-process publication and is outside this single-user design.

Default resource gates, configurable in settings:

- LLM runs: 4, matching the current llama.cpp slots;
- parser processes: 1, based on measured memory use;
- embedding requests: 1, because concurrent requests did not improve throughput;
- reranking requests: 1, for the same reason;
- bounded RAG CPU thread pool: 2 workers.

Large synchronous file and SQLite work must not run directly on the event loop. BM25 construction, vector preparation, and CPU ranking use the bounded thread pool. Parsing stays in a killable subprocess. Network calls use async clients.

Document embedding is split into configured batches and releases the embedding gate between batches. A waiting chat query can therefore proceed between ingestion batches instead of waiting for an entire document.

The UI naturally sends one turn at a time per session. The backend keeps only a defensive per-session active-run guard for duplicate clicks or direct API misuse. Different sessions, document APIs, and the document worker continue concurrently. No queue subsystem is introduced for normal chat turns.

## Consistency Invariants

The implementation must preserve these invariants:

1. A document is never externally observable as `ready` before a searchable snapshot containing it has been installed.
2. A failed ingestion never replaces the last valid snapshot.
3. A search sees exactly one complete snapshot.
4. Existing chunks are not re-embedded on ordinary add, delete, or restart.
5. A durable queued/running job is either completed, retried, failed, or recovered after restart; it is never silently lost.
6. A cancelled or failed chat run does not create a partial durable turn.
7. Session history never crosses session IDs.
8. Document data is shared across all sessions and remains independent from chat state.
9. A tool can only read ready documents present in its captured snapshot.
10. The event loop does not perform parser, BM25-build, large serialization, or other measured blocking work.

## UI Behavior

Keep the current single-page application and visual style. Change only the interaction structure required by the new model:

- a session list and a `New chat` action;
- the selected session's messages in the chat area;
- one active response state for the selected session;
- a document-management area independent from the message form;
- upload, list, download, retry, and delete controls;
- `processing`, `ready`, `failed`, and `deleting` status display.

While any document is non-terminal, poll `GET /api/documents` at a modest interval and stop polling when all documents are terminal. Do not add a document-status WebSocket or SSE channel.

Uploading never sends a message and never interrupts chat. Switching sessions within the page does not abort an existing fetch stream for another session; when a session is selected, the UI refreshes its persisted messages. Page unload or an actual network disconnect may cancel the associated run, and no stream-reconnection protocol is required. The initial implementation does not need multiple simultaneous visible streams in one browser tab, but the backend supports concurrent sessions.

## Failure Handling

- Convert expected validation and domain failures into typed application errors at module boundaries.
- Catch broad exceptions only at the job and HTTP/stream boundaries, where they can be logged and converted into a durable failure or SSE error.
- Preserve the last valid RAG snapshot on all preparation failures.
- Store only bounded, sanitized error text in SQLite and API responses.
- On shutdown, stop claiming new jobs, cancel active streams, terminate an active parser using the existing grace policy, close clients and sessions, and leave any interrupted durable job recoverable.
- Validate model responses: embedding row count/dimension and finite values, reranker result count and finite scores, non-empty final agent answer, and bounded tool results.
- Treat semantically empty parser output as failure even if the parser returned syntactically valid Markdown.

## Migration from Existing Data

Migration runs only when the new database is uninitialized and is idempotent through a recorded marker.

1. Import `data/corpus/corpus.json` document metadata and chunks.
2. Preserve existing overviews and source file IDs.
3. Mark imported documents `processing` and enqueue `reindex` jobs because the old JSON format has no embeddings.
4. Build and publish the first ready snapshot as reindex work succeeds.
5. Import `data/history/chat_history.json` user/assistant messages into one default Agents SDK session and create matching session metadata.
6. Keep both JSON files unchanged as recovery backups.

Malformed legacy records are reported explicitly and do not cause valid records or source files to be deleted. Migration tests cover absence, partial legacy data, repeated startup, and malformed input.

## Testing Strategy

### Unit tests

- schema migration and transaction rollback;
- job-state transitions, retry classification, and startup recovery;
- upload/path validation and orphan reconciliation;
- session context selection at item and character limits;
- agent tool schemas, file resolution, bounded results, and citation metadata;
- immutable snapshot add/delete/restart behavior;
- embedding persistence and embedding-signature invalidation;
- parser-output semantic validation.

### Integration tests

- all session and document HTTP contracts;
- SSE success, tool use, error, cancellation, and disconnect behavior;
- cancelled/failed runs leave no partial durable turn;
- upload returns before parsing completes;
- restart recovers queued and running jobs;
- deletion during ingestion cannot republish the deleted document;
- migration from the existing JSON corpus and history;
- clean application shutdown with an active parser or stream.

### Concurrency tests

- four different sessions can run without the old global `409` behavior;
- a duplicate request for the same session is rejected defensively;
- chat remains responsive while upload, parsing, and indexing proceed;
- a search using an old snapshot completes while add/delete publishes a new one;
- an event-loop ticker continues during 5,000-chunk BM25 construction and persistence work;
- ingestion batches release the embedding gate so a chat query is not starved.

### Agent quality and live tests

Retain and expand the existing agent fixture set. Continue measuring:

- tool-choice accuracy;
- follow-up document resolution;
- grounded-answer and citation correctness;
- empty or unsupported claims;
- latency to first visible content and total completion time;
- one-run versus four-run throughput.

Live llama.cpp tests remain opt-in. Unit and integration tests use deterministic fakes and run without model containers.

## Acceptance Criteria

- Existing supported document types, download, deletion, hybrid search, overview, citations, streaming, cancellation, and chat history remain functional.
- A document upload immediately returns a durable `processing` record and chat can continue while the job runs.
- Restart does not lose a document job or require re-embedding unchanged ready chunks.
- Multiple session chat runs and document work can overlap without an application-wide lock.
- Agents SDK owns the model/tool loop and session continuation; the custom loop and JSON history are gone.
- The model sees a bounded, coherent session suffix rather than unbounded history.
- Search never observes a partially built index and ingestion failure never damages the current index.
- The normal test suite passes, live agent evaluation does not regress from the current passing baseline, and concurrency probes show no event-loop stall from known CPU/persistence operations.
- No Redis, Celery, ORM, vector database, multi-agent layer, or speculative abstraction is added.

## References

- [OpenAI Agents SDK: Running agents](https://developers.openai.com/api/docs/guides/agents/running-agents)
- [OpenAI Agents SDK: Models and providers](https://developers.openai.com/api/docs/guides/agents/models)
