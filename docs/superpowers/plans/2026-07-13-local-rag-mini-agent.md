# Local RAG Mini-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace the legacy in-process model architecture with a minimal HTTP-based RAG mini-agent while retaining the current vanilla UI.

**Architecture:** FastAPI owns document persistence, Docling conversion, an in-memory BM25/vector index, and a one-action agent loop. A shared HTTPX async client calls independent llama.cpp LLM, embedding, and reranking servers. The agent plans one read-only action, receives its result in request memory, then streams the final answer; history never receives retrieved context or tool protocol.

**Tech Stack:** Python 3.12, FastAPI, HTTPX, Docling, BM25S, NumPy, Torch (Docling CUDA cleanup only), vanilla HTML/CSS/JS, pytest.

## Global Constraints

- Use test.py as the Docling conversion/chunking baseline and test.txt as the model-server API baseline.
- Do not import llama_cpp, use llama-cpp-python, load GGUFs, or free LLM/embed/reranker CUDA memory in the app.
- Use one shared HTTPX async client and raw HTTP for every model endpoint, including LLM SSE.
- Call torch.cuda.empty_cache() only in the finally path after Docling processing; do not synchronize or clear CUDA on stop, delete, clear-chat, startup, or normal chat.
- Persist only {role, content} for user/assistant history. RAG chunks, tool calls/results, scores, system prompt, and file tags are request-local.
- Documents are chat-scoped: restart rebuilds their index; clear chat deletes history, corpus, and copied uploads.
- Retain src/templates/index.html, src/static/style.css, and the current UI/UX. Do not add a frontend framework or build system.
- Keep commits concise, stage only task files, and never add Codex/AI/agent identity or co-author trailers.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| src/config.py | Settings, paths, model URLs, limits |
| src/models.py | Chunk, Document, Corpus, Message DTOs and JSON persistence |
| src/llama.py | HTTPX calls for planning, summary, answer SSE, embedding, reranking |
| src/rag.py | BM25, NumPy vectors, batch indexing, hybrid retrieval/rerank |
| src/documents.py | Safe upload copy, test.py Docling conversion, deletion/clear |
| src/chat.py | Three agent actions, validation/normalization, prompts, tool loop |
| src/main.py | Lifespan startup rebuild, app state, thin API and SSE routes |
| tests/ | Unit/API/static-UI tests with fake HTTP/Docling dependencies |

Delete after equivalent tests pass:

~~~
src/api/
src/core/
src/services/
~~~

### Task 1: Establish test harness and dependencies

**Files:**

- Modify: pyproject.toml
- Modify: uv.lock
- Create: tests/test_ui_assets.py

**Interfaces:**

- Produces pytest environment and direct dependencies used by the replacement modules.
- Does not modify the running application.

- [ ] **Step 1: Write failing static UI preservation tests**

~~~python
# tests/test_ui_assets.py
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_existing_ui_shell_and_controls_are_retained():
    html = (ROOT / "src/templates/index.html").read_text()
    for selector_id in (
        "prompt-form", "prompt-input", "file-input", "documents-list",
        "stop-response-btn", "delete-chats-btn", "toggle-sidebar-btn",
    ):
        assert f'id="{selector_id}"' in html

def test_browser_renders_server_content_as_text():
    script = (ROOT / "src/static/script.js").read_text()
    assert "textContent" in script
    assert "innerHTML = msg.content" not in script
~~~

- [ ] **Step 2: Run test to verify baseline**

Run: uv run pytest tests/test_ui_assets.py -v

Expected: PASS for the existing UI shell; any failure documents an asset that must be retained.

- [ ] **Step 3: Declare direct runtime/test dependencies**

~~~toml
[project]
name = "chatbot"
version = "3.0.0"
description = "Local RAG chatbot with a llama.cpp mini-agent"
requires-python = "==3.12.*"
dependencies = [
    "bm25s>=0.3.9",
    "docling>=2.112.0",
    "fastapi[standard]>=0.139.0",
    "httpx>=0.28.0",
    "numpy>=2.0.0",
    "torch>=2.0.0",
]

[dependency-groups]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0"]
~~~

Do not add openai, llama-cpp-python, a vector database, or an agent framework.

- [ ] **Step 4: Resolve environment**

Run: uv lock && uv sync --group dev

Expected: lockfile has HTTPX, NumPy, Torch, and pytest; no llama-cpp-python remains.

- [ ] **Step 5: Run harness**

Run: uv run pytest tests/test_ui_assets.py -v

Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add pyproject.toml uv.lock tests/test_ui_assets.py
git commit -m "test: add RAG harness"
~~~

### Task 2: Replace core persistence with DTOs

**Files:**

- Create: src/config.py
- Create: src/models.py
- Create: tests/test_models.py
- Delete: src/core/config.py
- Delete: src/core/models.py
- Delete: src/core/__init__.py

**Interfaces:**

- Settings exposes paths, llama server URLs, batch/candidate limits, and ensure_dirs().
- Corpus.load(path) accepts both new documents and legacy summaries keys.
- History.load(path) returns only user/assistant Message values.

- [ ] **Step 1: Write failing persistence/migration tests**

~~~python
# tests/test_models.py
from src.models import Chunk, Corpus, Document, History, Message

def test_corpus_round_trip_and_legacy_summary_migration(tmp_path):
    path = tmp_path / "corpus.json"
    corpus = Corpus([Document("doc-a", "rules.pdf", "summary", 1)], [
        Chunk("doc-a", "rules.pdf", 0, ["#/texts/1"], "deadline")
    ])
    corpus.save(path)
    assert Corpus.load(path) == corpus
    path.write_text('{"summaries":[{"file_id":"old","file_name":"old.pdf","summary":"s","chunk_count":0}],"chunks":[]}')
    assert Corpus.load(path).documents[0].file_id == "old"

def test_history_keeps_only_clean_turns(tmp_path):
    path = tmp_path / "history.json"
    path.write_text('{"messages":[{"role":"system","content":"old"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello","rag_context":{"x":1}}]}')
    assert History.load(path).messages == [Message("user", "hi"), Message("assistant", "hello")]
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: uv run pytest tests/test_models.py -v

Expected: FAIL with missing src.models.

- [ ] **Step 3: Implement DTOs**

~~~python
# src/config.py
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    uploads_dir: Path = data_dir / "uploads"
    corpus_path: Path = data_dir / "corpus" / "corpus.json"
    history_path: Path = data_dir / "history" / "chat_history.json"
    llm_url: str = os.getenv("LLM_URL", "http://127.0.0.1:8080")
    embed_url: str = os.getenv("EMBED_URL", "http://127.0.0.1:8081")
    rerank_url: str = os.getenv("RERANK_URL", "http://127.0.0.1:8082")
    llm_model: str = os.getenv("LLM_MODEL", "/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")
    embed_batch_size: int = 32
    candidate_limit: int = 16
    default_chunk_limit: int = 4

    def ensure_dirs(self) -> None:
        for path in (self.uploads_dir, self.corpus_path.parent, self.history_path.parent):
            path.mkdir(parents=True, exist_ok=True)
~~~

~~~python
# src/models.py: public types
@dataclass(frozen=True)
class Chunk:
    file_id: str
    file_name: str
    chunk_id: int
    refs: list[str]
    text: str
    # to_dict() and from_dict() preserve every field

@dataclass(frozen=True)
class Document:
    file_id: str
    file_name: str
    summary: str
    chunk_count: int
    # to_dict() and from_dict() preserve every field

@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str

@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    # to_dict(), save(path), load(path), with_document(), without_document()

@dataclass
class History:
    messages: list[Message] = field(default_factory=list)
    # load(path), save(path), append_turn(user, assistant), clear()
~~~

Corpus.load uses data.get("documents", data.get("summaries", [])); History.load filters roles to user and assistant and rewrites the clean file.

- [ ] **Step 4: Run DTO tests**

Run: uv run pytest tests/test_models.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/config.py src/models.py tests/test_models.py
git rm src/core/config.py src/core/models.py src/core/__init__.py
git commit -m "feat: add clean DTOs"
~~~

### Task 3: Add the shared llama.cpp HTTP client

**Files:**

- Create: src/llama.py
- Create: tests/test_llama.py

**Interfaces:**

- LlamaClient exposes embed(texts), rerank(query, documents), plan(messages, tools), summarize(file_name, chunks), and stream_answer(messages).
- stream_answer yields only nonempty text deltas.
- Malformed/non-2xx model responses raise ModelHTTPError.

- [ ] **Step 1: Write failing MockTransport tests**

~~~python
# tests/test_llama.py
import httpx
import json
import pytest
from src.llama import LlamaClient

@pytest.mark.asyncio
async def test_embed_posts_batch_and_returns_vectors():
    async def handler(request):
        assert request.url.path == "/embedding"
        assert json.loads(request.content) == {"content": ["a", "b"]}
        return httpx.Response(200, json=[{"index": 0, "embedding": [[1.0, 0.0], [0.0, 1.0]]}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        vectors = await LlamaClient(http, "http://llm", "http://embed", "http://rerank", "model").embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]

@pytest.mark.asyncio
async def test_stream_answer_yields_sse_deltas():
    body = 'data: {"choices":[{"delta":{"content":"Xin "}}]}\n\ndata: {"choices":[{"delta":{"content":"chào"}}]}\n\ndata: [DONE]\n\n'
    async def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = [x async for x in LlamaClient(http, "http://llm", "http://embed", "http://rerank", "model").stream_answer([])]
    assert result == ["Xin ", "chào"]
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: uv run pytest tests/test_llama.py -v

Expected: FAIL with missing src.llama.

- [ ] **Step 3: Implement raw HTTP methods**

~~~python
class ModelHTTPError(RuntimeError):
    pass

class LlamaClient:
    def __init__(self, client, llm_url, embed_url, rerank_url, model): ...

    async def embed(self, texts):
        response = await self.client.post(self.embed_url + "/embedding", json={"content": texts})
        response.raise_for_status()
        return response.json()[0]["embedding"]

    async def rerank(self, query, documents):
        response = await self.client.post(self.rerank_url + "/reranking", json={"query": query, "documents": documents})
        response.raise_for_status()
        return [item["relevance_score"] for item in response.json()["results"]]

    async def plan(self, messages, tools):
        return await self._chat({"messages": messages, "tools": tools, "tool_choice": "required", "temperature": 0, "max_tokens": 128})

    async def stream_answer(self, messages):
        payload = {"model": self.model, "messages": messages, "stream": True, "temperature": 0.1}
        async with self.client.stream("POST", self.llm_url + "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    content = json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        yield content
~~~

_chat adds the model, validates the response shape, and returns choices[0].message. summarize uses nonstream chat with a fixed Vietnamese summary/outline/key-points prompt.

- [ ] **Step 4: Run HTTP tests**

Run: uv run pytest tests/test_llama.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/llama.py tests/test_llama.py
git commit -m "feat: add llama HTTP client"
~~~

### Task 4: Implement compact hybrid retrieval and startup rebuild

**Files:**

- Create: src/rag.py
- Create: tests/conftest.py
- Create: tests/test_rag.py

**Interfaces:**

- RagIndex(llama, batch_size, candidate_limit) exposes rebuild(corpus), add(chunks), remove_document(file_id), and search(queries, file_ids, limit).
- rebuild embeds all chunks in batches; add embeds only supplied chunks.
- search returns ordered Chunk values without mutating Corpus.

- [ ] **Step 1: Add the shared corpus factory**

~~~python
# tests/conftest.py
import pytest
from src.models import Chunk, Corpus, Document

@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        documents=[Document("doc-a", "rules.pdf", "Deadline and submission rules.", 2)],
        chunks=[
            Chunk("doc-a", "rules.pdf", 0, ["#/texts/1"], "Submit before Friday."),
            Chunk("doc-a", "rules.pdf", 1, ["#/texts/2"], "Late submissions need approval."),
        ],
    )
~~~

- [ ] **Step 2: Write failing retrieval tests**

~~~python
# tests/test_rag.py
import pytest
from src.rag import RagIndex

class FakeLlama:
    def __init__(self):
        self.embed_calls = []
    async def embed(self, texts):
        self.embed_calls.append(texts)
        return [[float(i == 0), float(i == 1)] for i, _ in enumerate(texts)]
    async def rerank(self, query, documents):
        return [float(len(doc)) for doc in documents]

@pytest.mark.asyncio
async def test_add_embeds_only_new_chunks_and_filters_documents(corpus):
    llama = FakeLlama()
    index = RagIndex(llama, 32, 16)
    await index.rebuild(corpus)
    await index.add([])
    result = await index.search(["deadline"], ["doc-a"], 1)
    assert len(llama.embed_calls) == 2
    assert [chunk.file_id for chunk in result] == ["doc-a"]

@pytest.mark.asyncio
async def test_remove_document_filters_vectors_and_chunks(corpus):
    index = RagIndex(FakeLlama(), 32, 16)
    await index.rebuild(corpus)
    index.remove_document("doc-a")
    assert index.chunk_count == 0
~~~

- [ ] **Step 3: Run test to verify it fails**

Run: uv run pytest tests/test_rag.py -v

Expected: FAIL with missing src.rag.

- [ ] **Step 4: Implement local indexing/retrieval**

~~~python
class RagIndex:
    async def rebuild(self, corpus):
        self.chunks = list(corpus.chunks)
        self.vectors = await self._embed_batched([chunk.text for chunk in self.chunks])
        self._build_bm25()

    async def add(self, chunks):
        if not chunks:
            return
        new_vectors = await self._embed_batched([chunk.text for chunk in chunks])
        self.chunks.extend(chunks)
        self.vectors = np.vstack((self.vectors, new_vectors)) if len(self.vectors) else new_vectors
        self._build_bm25()

    async def search(self, queries, file_ids, limit):
        allowed = {i for i, chunk in enumerate(self.chunks) if not file_ids or chunk.file_id in file_ids}
        best_scores = {}
        for query in queries:
            candidates = self._fuse_candidates(query, allowed)
            scores = await self.llama.rerank(query, [self.chunks[i].text for i in candidates])
            for index, score in zip(candidates, scores):
                best_scores[index] = max(best_scores.get(index, float("-inf")), score)
        ordered = sorted(best_scores, key=best_scores.__getitem__, reverse=True)
        return [self.chunks[index] for index in ordered[:limit]]
~~~

Implement _fuse_candidates with local BM25 plus normalized dot product and reciprocal rank 1/(60+rank). For multi-query union candidates, rerank each query's bounded candidate list, and retain each candidate's best score. Cap candidate texts at 16. Do not use a vector-store framework.

- [ ] **Step 5: Run retrieval tests**

Run: uv run pytest tests/test_rag.py -v

Expected: PASS.

- [ ] **Step 6: Commit**

~~~bash
git add src/rag.py tests/conftest.py tests/test_rag.py
git commit -m "feat: add hybrid retrieval"
~~~

### Task 5: Implement Docling upload transaction and cleanup

**Files:**

- Create: src/documents.py
- Create: tests/test_documents.py

**Interfaces:**

- DocumentService(settings, llama, rag, corpus) exposes ingest(upload_name, content, cancelled), delete(file_id), and clear().
- ingest returns Document and commits corpus/index only after copy, conversion, summary, and embedding succeed.
- convert_document follows test.py and releases Docling/CUDA exactly in finally.

- [ ] **Step 1: Write failing transaction tests**

~~~python
# tests/test_documents.py
import asyncio
import pytest
from src.documents import DocumentService

@pytest.mark.asyncio
async def test_failed_summary_rolls_back_copy(tmp_path, monkeypatch):
    service = DocumentService.for_test(tmp_path)
    monkeypatch.setattr(service, "convert_document", lambda *_: [])
    async def fail(*_):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(service.llama, "summarize", fail)
    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await service.ingest("../../unsafe.pdf", b"pdf", asyncio.Event())
    assert list(service.settings.uploads_dir.iterdir()) == []
    assert service.corpus.documents == []

def test_upload_name_is_not_a_path(tmp_path):
    assert DocumentService.for_test(tmp_path).safe_name("../../unsafe.pdf") == "unsafe.pdf"
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: uv run pytest tests/test_documents.py -v

Expected: FAIL with missing src.documents.

- [ ] **Step 3: Implement test.py Docling path and transaction**

~~~python
def convert_document(file_name, content):
    converter = chunker = None
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import DocumentStream, InputFormat
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        pipeline = PdfPipelineOptions()
        pipeline.do_ocr = False
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline, backend=PyPdfiumDocumentBackend,
            )},
        )
        document = converter.convert(DocumentStream(name=file_name, stream=BytesIO(content))).document
        chunker = build_markdown_chunker()
        return chunks_from_doc(document, file_name, chunker)
    finally:
        del converter, chunker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
~~~

chunks_from_doc uses the markdown table serializer, HybridChunker, DocChunk refs, and contextualize call from test.py. ingest holds one asyncio.Lock, writes <id>_<safe_name>, runs conversion with asyncio.to_thread, checks cancelled.is_set before commit, awaits summary and rag.add(chunks), then saves the updated corpus. On error/cancel it removes the copy and restores the prior corpus/index snapshot. Do not preload Docling or use an executor.

- [ ] **Step 4: Run document tests**

Run: uv run pytest tests/test_documents.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/documents.py tests/test_documents.py
git commit -m "feat: add document ingestion"
~~~

### Task 6: Implement the three-action mini-agent

**Files:**

- Create: src/chat.py
- Create: tests/test_chat.py

**Interfaces:**

- ChatService(llama, rag, corpus, history) exposes stream(message, new_document, cancelled).
- parse_action(tool_call, corpus) returns a validated AgentAction or answer/clarify.
- Planner gets only catalog plus recent clean history; final request gets tool result/context; only final user/assistant is saved.

- [ ] **Step 1: Write failing agent tests**

~~~python
# tests/test_chat.py
import pytest
from src.chat import ChatService, parse_action

def test_unique_filename_normalizes_but_unknown_clarifies(corpus):
    action = parse_action({"function": {"name": "get_summaries", "arguments": '{"file_ids":["rules.pdf"]}'}}, corpus)
    assert action.arguments["file_ids"] == ["doc-a"]
    unknown = parse_action({"function": {"name": "get_summaries", "arguments": '{"file_ids":["missing.pdf"]}'}}, corpus)
    assert unknown.name == "answer"
    assert unknown.arguments == {"mode": "clarify"}

@pytest.mark.asyncio
async def test_context_is_request_only(corpus, tmp_path):
    service = ChatService.with_fake_dependencies(corpus, tmp_path)
    events = [event async for event in service.stream("Khi nào nộp?", None, None)]
    assert any("content" in event for event in events)
    assert "Submit before Friday" not in service.history.messages[-1].content
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: uv run pytest tests/test_chat.py -v

Expected: FAIL with missing src.chat.

- [ ] **Step 3: Implement schemas, validation, and one action loop**

~~~python
TOOLS = [
    {"type": "function", "function": {"name": "answer", "parameters": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["normal", "acknowledge", "clarify"]}}, "required": ["mode"]}}},
    {"type": "function", "function": {"name": "get_summaries", "parameters": {"type": "object", "properties": {"file_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["file_ids"]}}},
    {"type": "function", "function": {"name": "search_documents", "parameters": {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}, "file_ids": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 6}}, "required": ["queries", "file_ids", "limit"]}}},
]

async def stream(self, message, new_document, cancelled):
    yield {"status": "Đang xác định cách hỗ trợ..."}
    plan_message = await self.llama.plan(self.planner_messages(message, new_document), TOOLS)
    action = parse_action(plan_message["tool_calls"][0], self.corpus)
    if action.name == "search_documents":
        chunks = await self.rag.search(**action.arguments)
        tool_result = {"chunks": [chunk.to_dict() for chunk in chunks]}
    elif action.name == "get_summaries":
        tool_result = {"documents": self.corpus.summaries_for(action.arguments["file_ids"])}
    else:
        tool_result = {"mode": action.arguments["mode"]}
    full = ""
    async for part in self.llama.stream_answer(self.answer_messages(message, plan_message, tool_result)):
        if cancelled and cancelled.is_set():
            yield {"cancelled": True}
            return
        full += part
        yield {"content": part}
    self.history.append_turn(message, full)
    self.history.save(self.history_path)
    yield {"done": True}
~~~

parse_action accepts only the three names, JSON-decodes arguments, caps three nonempty queries and limits 1..6, maps an exact unique filename to its opaque ID, and otherwise returns answer/clarify. answer_messages includes a role tool message with the planner tool-call ID and a source-citation instruction; it never adds that message or raw chunks to History.

- [ ] **Step 4: Run agent tests**

Run: uv run pytest tests/test_chat.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add src/chat.py tests/test_chat.py
git commit -m "feat: add document mini-agent"
~~~

### Task 7: Replace the API application and wire restart

**Files:**

- Modify: src/main.py
- Create: tests/test_api.py
- Delete: src/api/routes.py
- Delete: src/api/__init__.py
- Delete: src/core/cancellation.py

**Interfaces:**

- create_app(settings, build_services=...) constructs shared HTTP client, Corpus, History, RagIndex, DocumentService, and ChatService; tests pass a fake build_services function.
- Lifespan rebuilds the index before app readiness and closes HTTP client without CUDA cleanup.
- Existing API paths stay stable. stop accepts request_id and sets only that request event.

- [ ] **Step 1: Write failing FastAPI tests**

~~~python
# tests/test_api.py
import asyncio
from dataclasses import replace
from fastapi.testclient import TestClient
from src.config import settings
from src.main import create_app

def test_startup_rebuilds_existing_corpus(tmp_path, fake_build_services):
    app = create_app(replace(settings, data_dir=tmp_path), build_services=fake_build_services)
    with TestClient(app) as client:
        assert app.state.rag.rebuild_calls == 1
        assert client.get("/api/documents").json() == {"documents": []}

def test_stop_marks_only_its_request(tmp_path, fake_build_services):
    app = create_app(replace(settings, data_dir=tmp_path), build_services=fake_build_services)
    with TestClient(app) as client:
        app.state.cancel_events["one"] = asyncio.Event()
        assert client.post("/api/stop?request_id=one").json() == {"status": "ok"}
        assert app.state.cancel_events["one"].is_set()
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: uv run pytest tests/test_api.py -v

Expected: FAIL because the legacy app imports old services.

- [ ] **Step 3: Implement lifespan and thin API**

~~~python
@asynccontextmanager
async def lifespan(app):
    app.state.settings.ensure_dirs()
    app.state.corpus = Corpus.load(app.state.settings.corpus_path)
    app.state.history = History.load(app.state.settings.history_path)
    await app.state.rag.rebuild(app.state.corpus)
    yield
    await app.state.http.aclose()

@app.post("/api/chat")
async def chat(message: str = Form(...), file: UploadFile | None = File(None)):
    request_id, cancelled = secrets.token_urlsafe(12), asyncio.Event()
    app.state.cancel_events[request_id] = cancelled
    async def events():
        yield sse({"request_id": request_id})
        try:
            document = await app.state.documents.ingest(file.filename, await file.read(), cancelled) if file else None
            async for event in app.state.chat.stream(message, document, cancelled):
                yield sse(event)
        finally:
            app.state.cancel_events.pop(request_id, None)
    return StreamingResponse(events(), media_type="text/event-stream")
~~~

Implement download/delete/clear with DocumentService; clear persists empty corpus/history and never clears shared model CUDA. Keep static and root routes unchanged.

- [ ] **Step 4: Run API tests**

Run: uv run pytest tests/test_api.py -v

Expected: PASS.

- [ ] **Step 5: Delete obsolete routing/cancellation code and commit**

~~~bash
git add src/main.py tests/test_api.py
git rm -r src/api src/core/cancellation.py
git commit -m "feat: wire HTTP RAG API"
~~~

### Task 8: Adapt the existing browser controller and remove legacy services

**Files:**

- Modify: src/static/script.js
- Modify: tests/test_ui_assets.py
- Delete: src/services/
- Delete: remaining src/core/ files if any

**Interfaces:**

- Browser stores the initial SSE request_id and posts it to stop.
- Existing chat controls, sidebar, theme, document download/delete, and clean history rendering remain.

- [ ] **Step 1: Extend failing UI test**

~~~python
def test_ui_uses_request_scoped_stop_and_existing_document_controls():
    script = (ROOT / "src/static/script.js").read_text()
    assert "requestId" in script
    assert "/api/stop?request_id=" in script
    assert "download-doc-btn" in script
    assert "delete-doc-btn" in script
~~~

- [ ] **Step 2: Run it to verify failure**

Run: uv run pytest tests/test_ui_assets.py::test_ui_uses_request_scoped_stop_and_existing_document_controls -v

Expected: FAIL because the old stop request has no ID.

- [ ] **Step 3: Make the smallest JavaScript change**

~~~javascript
let requestId = null;

// In the existing SSE handler, before status/content:
if (data.request_id) {
  requestId = data.request_id;
  continue;
}

// In cleanupRequest:
requestId = null;

// In the existing stop handler:
if (requestId) {
  await fetch("/api/stop?request_id=" + encodeURIComponent(requestId), { method: "POST" });
}
~~~

Retain all existing DOM IDs/classes and current textContent rendering. Do not change layout/styling or document/history/clear endpoint names.

- [ ] **Step 4: Run focused UI/full tests**

Run: uv run pytest tests/test_ui_assets.py tests/test_models.py tests/test_llama.py tests/test_rag.py tests/test_documents.py tests/test_chat.py tests/test_api.py -v

Expected: PASS.

- [ ] **Step 5: Remove obsolete service implementation and commit**

~~~bash
git add src/static/script.js tests/test_ui_assets.py
git rm -r src/services
git commit -m "refactor: remove local services"
~~~

### Task 9: Add E4B live capability/evaluation tests

**Files:**

- Create: tests/fixtures/agent_cases.json
- Create: tests/test_agent_eval.py

**Interfaces:**

- Fixture entries contain catalog, history, message, expected action, and expected file IDs.
- Live test runs only with RUN_LIVE_MODEL_TEST=1; normal CI has no model requirement.

- [ ] **Step 1: Write failing opt-in evaluation**

~~~json
[
  {"name":"upload acknowledgement","message":"Mình gửi file này để bạn đọc trước nhé.","new_document":"doc-a","action":"answer"},
  {"name":"existing detail","message":"Trong rules.pdf, hạn nộp là khi nào?","action":"search_documents","file_ids":["doc-a"]},
  {"name":"follow up","history":["Hạn nộp là thứ Sáu. Nguồn: rules.pdf, đoạn 0"],"message":"Ý đó có được nộp muộn không?","action":"search_documents","file_ids":["doc-a"]},
  {"name":"small talk","message":"Chào bạn","action":"answer"}
]
~~~

~~~python
@pytest.mark.skipif(os.getenv("RUN_LIVE_MODEL_TEST") != "1", reason="requires local llama.cpp")
@pytest.mark.asyncio
async def test_e4b_agent_cases(agent_cases, live_chat_service):
    for case in agent_cases:
        action = await live_chat_service.plan_only(**case)
        assert action.name == case["action"], case["name"]
        if "file_ids" in case:
            assert action.arguments["file_ids"] == case["file_ids"], case["name"]
~~~

- [ ] **Step 2: Verify normal runs skip the live test**

Run: uv run pytest tests/test_agent_eval.py -v

Expected: SKIPPED with requires local llama.cpp.

- [ ] **Step 3: Expand to 40-60 Vietnamese cases**

Cover upload acknowledgement/summary, summary/main-points/structure, exact/all/multiple file questions, clean-history follow-ups, ambiguous/unknown filenames, and small talk. Store only expected action/permitted IDs, never raw sensitive content.

- [ ] **Step 4: Run live E4B evaluation**

Run: RUN_LIVE_MODEL_TEST=1 uv run pytest tests/test_agent_eval.py -v

Expected: valid tool calls; at least 95% correct action selection and at least 90% correct follow-up document selection. Improve prompt/schema before changing models.

- [ ] **Step 5: Run full verification**

Run: uv run pytest -v

Expected: model-free suite PASS; live suite PASS when enabled.

- [ ] **Step 6: Commit**

~~~bash
git add tests/fixtures/agent_cases.json tests/test_agent_eval.py
git commit -m "test: add agent evaluation"
~~~

## Final Verification

- [ ] Run uv run pytest -v and record the complete passing output.
- [ ] Start the three containers from test.txt; start FastAPI against a corpus containing a copied upload; confirm restart rebuilds BM25/vectors before ready.
- [ ] In retained UI: upload PDF/DOCX, acknowledge and summarize a just-uploaded file, ask detailed/follow-up questions, download/delete, stop a stream, clear chat, restart FastAPI, and confirm document/history behavior.
- [ ] Confirm rg -n "llama_cpp|LlamaEmbedding|empty_cache|cuda.synchronize|ThreadPoolExecutor" src pyproject.toml returns only deliberate Docling torch.cuda.empty_cache cleanup and no legacy model lifecycle code.
- [ ] Confirm git status --short contains no accidental changes. Do not stage current user changes test.txt, test.py, test.json, dependency experiments, or README deletion unless explicitly brought into scope.
