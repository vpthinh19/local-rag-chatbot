# Local RAG Mini-Agent Design

## 1. Goal

Replace the legacy in-process `llama-cpp-python` backend with a small local RAG application whose LLM is the center of a constrained document agent.

The application keeps only the concepts that remain useful: explicit DTOs, a persisted corpus, document overviews, hybrid retrieval, clean chat history, and the existing browser UI. The legacy service hierarchy, semantic analyzer, cancellation framework, executors, and model lifecycle code are not migration targets.

The result is a single-user pet project. Simplicity, predictable cleanup, and readable control flow take priority over extensibility or distributed-system patterns.

## 2. Scope and non-goals

Included:

- PDF and DOCX ingestion with the Docling conversion/chunking behavior proven in `test.py`.
- A fresh Docling subprocess for each upload, terminated after contextualized chunks have been produced.
- Three independently hosted `llama.cpp` HTTP services for generation, embeddings, and reranking.
- A read-only LLM agent that either answers directly or calls at most one document tool.
- BM25 plus embedding retrieval, reciprocal-rank fusion, and bounded reranking.
- Chat-scoped document persistence, download, delete, clear-chat, stop, and restart rebuild.
- The existing vanilla HTML/CSS/JavaScript UI with minimal compatibility and safety fixes.

Excluded:

- Multiple users, accounts, tenants, or simultaneous chat requests.
- Redis, Celery, a persistent job queue, a database, or a vector database.
- Agent frameworks, workflow graphs, generic tool registries, or open-ended tool loops.
- Web search, arbitrary code execution, filesystem tools, or destructive LLM tools.
- Persistent embedding vectors in the first version.
- Resuming a chat generation after a browser disconnect.

## 3. Runtime process model

The normal system has four persistent processes:

| Process | Endpoint | Responsibility |
| --- | --- | --- |
| FastAPI, one Uvicorn worker | `:8000` | UI/API/SSE, orchestration, persistence, BM25, vectors, agent harness |
| LLM `llama.cpp` server | `:8080/v1/chat/completions` | Direct answers, tool selection, summaries, final grounded answers |
| Embedding `llama.cpp` server | `:8081/embedding` | Batch document and query embeddings |
| Reranking `llama.cpp` server | `:8082/reranking` | Score bounded candidates for a query |

During document conversion there is one additional temporary process:

| Process | Lifetime | Responsibility |
| --- | --- | --- |
| Docling worker | One upload, from convert through contextualized chunk output | Read PDF/DOCX, convert, chunk, preserve refs, write plain JSON result, exit |

The app must run with one Uvicorn worker. Multiple workers would duplicate the in-memory corpus/index and cancellation state.

File copying, JSON persistence, BM25, NumPy similarity, fusion, and HTTP orchestration remain in FastAPI. They are not separate processes.

## 4. Minimal source structure

```text
src/
  main.py             # FastAPI lifespan, one active request, endpoints, SSE
  config.py           # paths, service URLs, timeouts, retrieval limits
  models.py           # DTOs and atomic corpus/history JSON persistence
  llama.py            # one shared HTTPX client and llama.cpp protocols
  rag.py              # BM25, normalized vectors, fusion, reranking
  docling_worker.py   # subprocess entrypoint: convert and chunk one file
  documents.py        # staging, worker lifecycle, ingest transaction, delete/clear
  chat.py             # prompts, two tools, validation, one-tool agent loop
```

Keep functions and concrete classes small. Do not introduce base services, repositories, dependency-injection containers, event buses, or generic workflow abstractions.

## 5. Data model and persistence

Retain explicit boundary DTOs:

- `Chunk(file_id, file_name, chunk_id, refs, text)`
- `Document(file_id, file_name, overview, chunk_count)`
- `Corpus(documents, chunks)`
- `Message(role, content)` where role is only `user` or `assistant`
- `AgentToolCall(name, arguments, call_id)` as a request-local value

Paths:

```text
data/
  uploads/<file_id>_<safe_name>
  staging/<request_id>/input.<ext>
  staging/<request_id>/chunks.json
  corpus/corpus.json
  history/chat_history.json
```

`Corpus.load()` accepts the legacy `summaries` key and migrates it to `documents`. History loading discards legacy system messages and `rag_context` fields. Saving history stores only ordered `{role, content}` pairs.

JSON persistence uses a temporary file in the destination directory followed by an atomic replace. Retrieved chunks, tool messages, scores, prompts, and internal file tags are never persisted as history.

## 6. Single active request

The UI locks chat submission until the current request completes or is stopped. The backend independently enforces one active `/api/chat` pipeline and rejects another with HTTP 409.

The active request state contains only:

- a request ID;
- a cancellation event;
- the current pipeline task;
- the Docling subprocess handle when one exists.

There is no durable job registry or multi-job queue. SSE emits coarse real states and periodic heartbeats while a long PDF is processed.

## 7. Document ingestion

### 7.1 Staging and worker

1. Validate a nonempty `.pdf` or `.docx` basename and generate opaque request/document IDs.
2. Copy the upload into a request-scoped staging directory. Never form a path directly from an unsanitized client filename.
3. Spawn `python -m src.docling_worker` with explicit input, output, filename, and file-ID arguments. Do not use a shell.
4. Await the subprocess asynchronously so the FastAPI event loop remains responsive.
5. The worker follows `test.py`: `PdfPipelineOptions` with OCR off, `PyPdfiumDocumentBackend`, `MsWordDocumentBackend`, markdown-table serialization, `HybridChunker(merge_peers=True, always_emit_headings=True)`, refs from `DocChunk`, and contextualized text.
6. The worker writes only plain chunk JSON and exits. Its process exit destroys the Docling CUDA context and releases its VRAM before summary or embedding begins.

The FastAPI process does not import or initialize Docling conversion components. `torch.cuda.empty_cache()` in FastAPI is unnecessary. The worker may perform best-effort local cleanup, but process exit is the actual resource boundary.

### 7.2 Finalization and commit

After the worker exits successfully:

1. Validate the result JSON and convert it to `Chunk` DTOs.
2. Ask the LLM for one bounded structured overview covering summary, outline, and key points.
3. Embed only the new chunks in batches.
4. Build a candidate corpus and candidate RAG state without mutating the live state.
5. Enter a short non-awaiting commit section: move the staged upload to its final path, atomically save the candidate corpus, and swap the candidate corpus/index into live state.
6. Remove the staging directory.

If conversion, overview, embedding, validation, or persistence fails before commit, remove staging and retain the previous live/persisted corpus and index.

On process crash between moving the upload and saving corpus, startup orphan cleanup removes the unreferenced upload. On startup, documents whose final upload is missing are removed before rebuilding the index.

### 7.3 Delete and clear

Delete prepares corpus/index state first, temporarily moves the final upload out of its public path, atomically saves and installs the candidate state, then removes the temporary file. A save failure restores the upload and leaves live state untouched. Clear chat cancels the active request, persists empty corpus/history, installs empty state, and then removes committed uploads; startup cleanup handles any file that could not be removed.

Delete is rejected while a chat pipeline is active. Clear-chat is allowed to cancel that pipeline first. These operations never stop or clean the persistent llama.cpp servers.

## 8. Cancellation semantics

The rule is: discard work that has not committed; preserve state that has committed.

| Cancellation phase | Action | Persistent result |
| --- | --- | --- |
| Docling running | terminate, wait briefly, kill if needed, reap, remove staging | Existing corpus/history unchanged |
| Overview request | cancel/close HTTP request, remove staging | Existing corpus/history unchanged |
| Embedding request | cancel/close HTTP request, discard candidate state, remove staging | Existing corpus/history unchanged |
| Commit section | finish the short atomic commit, then stop | New document remains persisted |
| Agent planning/retrieval/generation | close active HTTP stream/request and stop SSE | Documents remain; incomplete chat turn is not saved |

Stopping never kills the LLM, embedding, or reranking server processes. A remote llama.cpp request may finish after its client disconnects, but its result is ignored.

Partial assistant responses and their user turn are not persisted. If ingestion committed before chat cancellation, the document remains visible after the UI refreshes its document list.

## 9. RAG index and retrieval

FastAPI owns:

- one BM25 index;
- one normalized `numpy.float32` embedding matrix;
- one ordered chunk list aligned with vector rows.

Startup loads/prunes corpus, rebuilds BM25, and embeds all persisted chunks in batches. A nonempty corpus is not considered ready if embedding rebuild fails or returns malformed data.

For `search_documents(queries, file_ids, limit)`:

1. Accept one to three nonempty queries, one to eight existing file IDs, and a final limit from one through six.
2. Embed all queries in one request.
3. For each query, collect bounded BM25 and cosine-similarity rankings restricted to selected files.
4. Fuse rankings locally with reciprocal rank.
5. Send at most 16 candidate texts per query to reranking.
6. Map reranking results by their explicit response `index`, validate unique/in-range indices, and combine a candidate's best score across queries.
7. Return the top chunks with filename, file ID, chunk ID, refs, and text.

Embedding responses must have the requested row count, one stable nonzero dimension, and finite numeric values. Reranking score direction and real model quality must be verified against the live service because the current `test.txt` example ranks an incorrect statement above the correct one.

Uploading embeds only new chunks. Deleting filters existing vector rows. BM25 rebuilds from the resulting chunk list.

## 10. LLM-centered mini-agent

The harness gives the LLM:

- a compact catalog of ready documents (`file_id`, filename, chunk count);
- recent clean user/assistant turns;
- the newly committed document ID when applicable;
- the current user message;
- two read-only tools.

The tools are:

```text
get_document_overviews(file_ids)
search_documents(queries, file_ids, limit)
```

The model may either answer directly or call exactly one tool. There is no artificial `answer(mode=...)` tool and no status-polling tool.

### 10.1 Direct-answer cases

- greetings and general conversation;
- upload acknowledgement when the message asks only to read/hold the file;
- questions about the assistant rather than document contents.

These complete in one streaming LLM call.

### 10.2 Overview cases

- summary, outline, structure, themes, or main points;
- high-level comparison of selected documents.

The LLM calls `get_document_overviews`, receives stored overviews, then produces the final streamed answer.

### 10.3 Search cases

- specific facts, definitions, explanations, extraction, or detailed comparisons;
- follow-ups whose standalone queries and document IDs can be inferred from recent clean history and citations.

The LLM calls `search_documents`, receives bounded chunks, then produces the final streamed answer with compact citations.

### 10.4 Harness behavior

The harness validates tool name, JSON shape, query count, limits, file IDs, and context size. An exact unique filename may normalize to its ID. Unknown or ambiguous references are not replaced with all documents.

A valid but unresolvable call returns a structured tool error so the final LLM response asks for clarification. Empty retrieval returns a structured `no_usable_results` result and the model must not claim a document fact. Malformed model protocol or HTTP responses produce a controlled error rather than an invented fallback operation.

The first chat-completion request uses native tools with `tool_choice="auto"` and streaming. Content deltas are forwarded as a direct answer. Tool-call deltas are accumulated and not shown to the user. After one validated tool result, the second request streams the final answer with tools disabled. A second tool call is rejected.

The high generation throughput of the QAT 4-bit model with MTP is used primarily for direct and final streaming. Keeping direct answers to one call, prompts compact, and the tool loop bounded matters more than adding planner calls.

## 11. HTTP client

FastAPI owns one long-lived `httpx.AsyncClient` with connection pooling and explicit connect, read, write, and pool timeouts. It calls the endpoints from `test.txt` directly; the OpenAI SDK is not required.

The client validates:

- non-2xx responses and useful bounded error text;
- chat response and streaming SSE shapes;
- streamed content versus tool-call deltas;
- embedding row/dimension/finite-value invariants;
- reranking indices and finite scores.

Closing or cancelling an app request closes its active HTTP response context. The shared `AsyncClient` itself remains alive until FastAPI shutdown.

## 12. API and UI contract

Retain these endpoints:

- `POST /api/chat`: message plus optional PDF/DOCX; SSE request/status/content/done/error/cancelled events.
- `POST /api/stop`: cancel the single active request.
- `GET /api/chat-history`: clean saved messages.
- `POST /api/clear-chat`: cancel active work and clear chat-scoped documents/history.
- `GET /api/documents`: committed document metadata only.
- `GET /api/documents/{file_id}/download`.
- `DELETE /api/documents/{file_id}`.

The existing UI locks submission while a request is active, shows coarse ingest/agent status, supports stop, refreshes documents after completion/cancel, and keeps its theme/sidebar behavior.

All server/user-derived content, including filenames and attachment names, is rendered with `textContent` or DOM properties. It is never interpolated into `innerHTML`.

## 13. Configuration and dependencies

Direct runtime dependencies:

- `fastapi[standard]`
- `docling`
- `bm25s`
- `numpy`
- `httpx`
- `torch` only insofar as Docling requires it

Development dependencies are `pytest` and `pytest-asyncio`. Do not add `llama-cpp-python`, OpenAI SDK, an agent framework, a task queue, or a vector database.

Configuration exposes paths; `LLM_URL`, `EMBED_URL`, and `RERANK_URL`; HTTP timeouts; embedding batch size; lexical/semantic/candidate/final limits; worker termination grace time; and maximum upload/context sizes.

Model file paths and CUDA flags belong only to the container commands, not application configuration.

## 14. Acceptance criteria

Automated tests cover:

1. DTO migration and atomic corpus/history persistence.
2. Safe upload names and staging cleanup.
3. Successful worker result, worker failure, terminate/kill cancellation, and no Docling import in FastAPI modules.
4. Rollback before commit and persistence after cancel during agent generation.
5. Startup orphan pruning and embedding rebuild.
6. Batch embeddings, filtered hybrid retrieval, reranking index mapping, and new-chunks-only embedding.
7. Direct answer, overview tool, search tool, invalid/ambiguous references, empty retrieval, and one-tool maximum.
8. Clean history and request-local tool/RAG context.
9. Single active request, SSE status/heartbeat, stop, disconnect cleanup, delete, clear, and download behavior.
10. Safe frontend rendering and retained UI controls.

A Vietnamese live fixture contains 40–60 representative cases. Targets are 100% valid tool protocol after allowed normalization, at least 95% correct direct/tool choice, at least 90% correct document selection in follow-ups, and zero unsupported document claims after empty retrieval.

Release verification includes a real PDF and DOCX ingest, observing that the Docling process exits and its VRAM is released before summary/embedding, cancellation in each major phase, restart rebuild, document operations, and live tool-call/final-answer behavior against the actual Gemma and BGE llama.cpp containers.

## 15. Final decisions

- Four persistent logical processes; one temporary Docling process only during ingestion.
- Exactly one active chat pipeline and one Uvicorn worker.
- No job queue or LLM progress tool.
- Subprocess exit, not `empty_cache()` in FastAPI, is the Docling VRAM boundary.
- Direct answer or at most one of two read-only tools.
- Uncommitted work is discarded on cancel; committed documents/history remain.
- Existing UI and JSON persistence are retained and cleaned up, not redesigned.
- The implementation stays concrete and small because this is a local pet project.
