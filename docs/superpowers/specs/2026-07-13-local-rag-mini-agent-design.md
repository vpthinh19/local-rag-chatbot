# Local RAG Mini-Agent Design

## Goal

Replace the legacy in-process `llama-cpp-python` architecture with a small FastAPI application that uses three independently running `llama.cpp` HTTP servers for generation, embeddings, and reranking. The application must retain a chat-scoped document library: uploaded files survive a server restart, but a user clearing chat deletes the history, indexed corpus, and copied uploads.

The LLM is a small, read-only document agent. It chooses a single action for each user turn, receives the action result, and then streams the user-facing answer. Retrieved context is request-scoped and must never be saved as chat history.

## Scope

Included:

- Process PDF and DOCX uploads with the Docling configuration and chunking behavior proven in `test.py`.
- Copy every accepted upload to server storage so the existing UI can list, download, and delete it.
- Rebuild BM25 and embeddings from the persisted corpus at FastAPI startup.
- Use plain async HTTP calls to the LLM, embedding, and reranking `llama.cpp` services described in `test.txt`.
- Implement a minimal tool-calling agent for acknowledgement, summary, and document search.
- Preserve the existing simple browser chat, document list, download, delete, and clear-chat experience.

Excluded:

- A persistent cross-chat document library, user accounts, multi-tenant isolation, or external database.
- LLM permissions to upload, delete, download, or clear documents.
- Persisting embedding vectors to disk in the first version.
- Web search, external tools, background agents, and arbitrary model-executed code.

## Runtime Services

The deployment has four independently managed processes.

| Service | Endpoint | Responsibility |
| --- | --- | --- |
| FastAPI app | `:8000` | UI, upload/storage, Docling, corpus persistence, BM25, agent orchestration, SSE to browser |
| LLM `llama.cpp` server | `:8080/v1/chat/completions` | Tool planning and streaming final answers |
| embedding `llama.cpp` server | `:8081/embedding` | Batch document/query embeddings |
| reranking `llama.cpp` server | `:8082/reranking` | Rank a small candidate set against a query |

`test.txt` is the operational baseline for the three model containers. The app does not load GGUF models, own their CUDA lifecycle, or import `llama_cpp`.

The app owns one long-lived `httpx.AsyncClient` with connection pooling and explicit connect/read/write timeouts. It calls the nonstandard embedding and reranking endpoints directly and parses the LLM server's OpenAI-compatible chat-completions SSE directly. `openai` SDK is deliberately not added: it would not improve model streaming throughput and would still require raw HTTP for `/embedding` and `/reranking`.

## Minimal Backend Structure

Keep modules focused and remove the legacy service hierarchy, cancellation framework, executors, and model wrappers.

```text
src/
  main.py        # FastAPI app, lifespan, thin API endpoints and SSE translation
  config.py      # paths, model service URLs, timeouts, retrieval limits
  models.py      # DTOs and corpus/history JSON load/save
  documents.py   # safe upload copy, Docling conversion/chunking, deletion/clear
  rag.py         # BM25 index, vectors, HTTP embedding/reranking, retrieval
  chat.py        # agent prompts, tool validation/normalization, LLM planning/answering
```

DTOs are intentionally retained because they make persistence and boundary validation legible:

- `Chunk(file_id, file_name, chunk_id, refs, text)`
- `Document(file_id, file_name, summary, chunk_count)`
- `Corpus(documents, chunks)`
- `Message(role, content)`
- `AgentAction(name, arguments)` as a request-local validated DTO only

`Corpus` persists as JSON under `data/corpus/corpus.json`. History persists as only ordered `{role, content}` pairs under `data/history/chat_history.json`; a system prompt, raw chunks, retrieval scores, tool calls, tool results, and RAG context are not persisted as messages.

## Document Lifecycle

### Upload and ingest

1. Validate that the upload has a PDF or DOCX extension and a nonempty sanitized basename. Generate an opaque `file_id`; never construct an upload path directly from a client filename.
2. Copy bytes to `data/uploads/<file_id>_<safe_name>` before processing so a successfully ingested document is downloadable after restart.
3. Process a `DocumentStream` using the `DocumentConverter`, `PdfPipelineOptions`, `PyPdfiumDocumentBackend`, markdown-table serializer, and `HybridChunker` pattern from `test.py`. Preserve `refs` and contextualized chunk text.
4. In `finally`, release only the Docling converter/chunker and call `torch.cuda.empty_cache()` once. No LLM, embed, or reranker CUDA cleanup occurs in the app.
5. Generate one structured document summary/outline/key-points card through the LLM. This stored card supplies overview questions without re-reading all chunks.
6. Embed only the new chunks in a batch. Append vectors, chunks, and document DTO atomically in memory; rebuild BM25 from the complete text list.
7. Persist corpus JSON only after all steps succeed. On failure, remove the copied upload and leave the current in-memory corpus/index untouched.

The ingest request is serialized because Docling is GPU-heavy and corpus mutation must be atomic. Generation and ordinary retrieval must not share this global lock.

### Restart

FastAPI lifespan loads corpus JSON and verifies that each document's copied upload still exists. Orphaned document metadata/chunks are removed and persisted before indexing.

For a nonempty corpus, startup rebuilds BM25 and sends all persisted chunk texts to the embedding server in batches. Vectors remain RAM-only. FastAPI reports ready only after this succeeds. If the embedding service is unavailable or returns malformed vectors, startup fails loudly instead of serving unindexed RAG responses.

### Delete and clear

Deleting one document removes its copied file, document DTO, chunks, matching vector rows, then rebuilds BM25. Clearing chat cancels the active chat stream, clears messages, deletes every copied upload, empties corpus/vector/BM25 state, and persists both empty JSON files. It does not call model cleanup endpoints or free LLM/embed/reranker CUDA memory.

## Retrieval

`rag.py` holds three plain data structures: a BM25 index, `numpy.float32` normalized embedding matrix, and an index-to-chunk mapping.

For `search_documents(queries, file_ids, limit)`:

1. Validate at most three nonempty queries and a limit from 1 through 6.
2. Resolve requested document references to existing opaque IDs. An exact unique filename may be normalized to its ID for model interoperability; an unknown or ambiguous reference is rejected.
3. Send all queries as one embedding request. For each query, obtain top lexical and semantic candidates, restricted to the requested documents when present.
4. Fuse lexical and semantic rank lists with a compact local reciprocal-rank calculation. Union candidates across query variants.
5. Send only the bounded candidate texts (maximum 16) to `/reranking` per query; combine scores by each candidate's best query score; select `limit` chunks.
6. Return text and source labels (`file_id`, filename, chunk ID, refs) only to the current chat request.

No framework vector store, agent framework, or model-managed index is used. On upload only newly created chunks are embedded. On delete, filtering the vector matrix plus rebuilding BM25 is preferred over an abstraction layer.

## Mini-Agent

Each user turn receives a compact document catalog (`file_id`, filename, chunk count), the last few persisted message pairs, any freshly ingested document ID, and the current message. The catalog contains no raw chunk data and no entire summaries.

The agent must choose exactly one action in a non-streaming, deterministic planning call (`temperature=0`, target maximum 128 output tokens):

| Action | Arguments | Purpose |
| --- | --- | --- |
| `answer` | `mode: normal | acknowledge | clarify` | General chat, upload acknowledgement, or an ambiguity question with no document read |
| `get_summaries` | `file_ids: list[str]` | Overview, outline, main-points, and structure questions |
| `search_documents` | `queries: list[str]`, `file_ids: list[str]`, `limit: int` | Specific questions, comparison, explanation, and document follow-ups |

The model has no destructive or file-system tools. The application validates action name and JSON argument shapes, enforces caps, and normalizes an exact unique filename to an ID where necessary. Invalid actions or unresolved references produce a controlled `answer(mode="clarify")` path; the application must not invent or execute a fallback document operation.

Native OpenAI-style `tools`/`tool_calls` are preferred because the running Gemma 4 E4B model was verified against the local `llama.cpp` server. The required capability test is: one planner call returns `finish_reason="tool_calls"`; a subsequent `role="tool"` message produces a grounded natural-language final response. If a deployment's model/chat template fails this test, use a strict JSON command (`action`, `queries`, `file_ids`, `limit`) with the same validation and loop. Do not parse arbitrary prose or use regex as the primary protocol.

The planner does not stream. After one action result is appended request-locally as a tool message, the LLM streams the final response to FastAPI; FastAPI forwards content deltas to the browser as SSE. The final answer includes compact source citations such as `Nguồn: quy_dinh_nop_bai.docx, đoạn 7`. These citations are part of the normal answer and let later clean-history turns resolve references such as “ý đó”, without storing raw RAG context.

## Chat and File Cases

| User situation | Agent action and result |
| --- | --- |
| Upload with “đọc trước” | Ingest first, then `answer(acknowledge)` streams confirmation |
| Upload with “tóm tắt/ý chính” | Ingest first, then `get_summaries` uses the new document ID |
| Question about an existing file | `search_documents` constrained to that file |
| “Tóm tắt”, “cấu trúc”, “ý chính” | `get_summaries` for selected file(s), all files when not specified |
| Follow-up “ý đó nghĩa là gì?” | Planner reads recent clean messages/citations, rewrites a standalone query, then searches the referenced file |
| Greeting or general question | `answer(normal)` and no retrieval HTTP calls |

## Cancellation, Concurrency, and Errors

Use one request ID and an `asyncio.Event` for browser stop/disconnect. Cancellation stops forwarding SSE and closes the active HTTP stream context. It is cooperative: an already-submitted model or Docling operation may finish at its service, but its result must not commit state after cancellation.

Only ingest/corpus mutation is serialized. Separate user chats may issue HTTP calls concurrently; the `llama.cpp` server owns its own scheduling. An active chat being stopped must never release shared model resources.

HTTP failures have clear user-visible SSE errors:

- LLM unavailable: generation/planning error, history unchanged.
- Embed/rerank unavailable: search error, no fabricated document answer.
- Docling failure/cancel: upload rollback, corpus unchanged.
- Corpus persistence failure: retain previous in-memory/index state and report failure; do not advertise the file as uploaded.

## UI and API Contract

The existing vanilla UI is reused, not redesigned or replaced. `src/templates/index.html` remains the page shell, `src/static/style.css` remains the visual design and responsive/sidebar behavior, and `src/static/script.js` remains the browser controller. Do not introduce React, a build pipeline, a component library, or a new visual system.

The only frontend changes are minimal compatibility changes required by the new backend contract:

- Render clean persisted history rather than the legacy internal `[FILE]` message form.
- Keep the existing chat stream/status/stop controls and adapt its SSE event parsing only when event names or payload fields change.
- Keep the current document sidebar, attachment indication, download action, delete action, clear-chat action, theme toggle, and collapsed-sidebar state.
- Continue using `textContent` for message/document values from the server; do not interpolate filenames or response text into `innerHTML`.
- Refresh the existing document list after successful ingest, single-document delete, cancellation, and clear chat.

Retain the existing browser endpoints where practical:

- `POST /api/chat` accepts a message and optional PDF/DOCX and returns SSE statuses/content/done/error.
- `POST /api/stop` requests cancellation only; it does not release CUDA memory.
- `GET /api/chat-history` returns clean saved messages.
- `POST /api/clear-chat` clears history and every chat-scoped document.
- `GET /api/documents` lists document DTO metadata.
- `GET /api/documents/{file_id}/download` streams the server copy.
- `DELETE /api/documents/{file_id}` removes one document and its index data.

No preview, separate document-library UI, or UX redesign is in scope.

## Dependencies and Configuration

Direct dependencies should be only FastAPI standard extras, Docling, BM25S, NumPy, HTTPX, and Torch (Torch is retained solely for Docling CUDA cache cleanup). Remove direct `llama-cpp-python` and model-path configuration. Model locations move to container commands; app config contains only service base URLs and operational values.

Configuration must expose:

- `LLM_URL`, `EMBED_URL`, `RERANK_URL`
- connect/read/write/pool timeouts
- `EMBED_BATCH_SIZE`, lexical/semantic candidate limits, rerank candidate cap (16), final chunk limit (1–6)
- upload and persistence paths

## Version Control

Keep commits short and consistent with the repository's existing concise style. Stage only files belonging to the current task; never include unrelated worktree changes.

Do not add `Codex`, agent, AI, or assistant identity to commit authorship, commit messages, bodies, trailers, or `Co-authored-by` lines. Use the repository's configured human Git identity unchanged.

## Verification and Acceptance Criteria

Automated tests must cover:

1. Corpus/history DTO persistence and restart rebuild invocation.
2. Safe upload naming, rollback for Docling/summary/embed failure, single-file delete, and clear-chat cleanup.
3. Batch embedding append, filtered retrieval, candidate fusion, rerank ordering, and no full re-embedding on upload.
4. Agent action validation: valid opaque ID, unique filename normalization, unknown/ambiguous reference rejection, action/argument caps, and no destructive action.
5. Message construction proving raw context/tool messages are absent from persisted history and present only in the final-answer request.
6. SSE response/cancellation behavior and no CUDA cleanup on stop, delete, clear, or normal chat.
7. Browser smoke coverage confirming the existing chat form, streaming answer area, document sidebar, download/delete controls, clear chat, and theme/sidebar preferences still work without a UI redesign.

Add a Vietnamese agent-evaluation fixture of 40–60 cases covering upload acknowledgement, upload summary, existing-file questions, summaries, multiple-file questions, follow-ups, ambiguous references, and small talk. Initial acceptance thresholds are:

- 100% action JSON/tool-call validity after allowed unique-filename normalization.
- At least 95% correct action selection.
- At least 90% correct document selection on follow-up scenarios.
- No answer claims a document fact when retrieval returned no usable chunks.

Run a live capability smoke test against the actual Gemma 4 E4B `llama.cpp` container before release: it must emit a structured tool call and then form a grounded answer after a `role="tool"` response. The current development environment passed this test.

## Decisions

- FastAPI app restart rebuilds BM25 and vectors from persisted corpus; no disk vector cache initially.
- Documents are chat-scoped and removed by clear chat.
- `httpx`, not OpenAI SDK, is the only model HTTP client.
- Exactly three agent actions are exposed; only two read document data.
- E4B is the current model. Keep a fixture-based evaluation so a future model change is evidence-driven rather than subjective.
- The existing vanilla HTML/CSS/JS UI is retained; only minimal contract and safe-rendering adjustments are allowed.
