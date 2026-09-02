# Local RAG Chatbot

Ứng dụng RAG cục bộ cho một người dùng: FastAPI, OpenAI Agents SDK, SQLite và ba
dịch vụ llama.cpp. Chạy **đúng một** ASGI worker; snapshot RAG nằm trong bộ nhớ
của process nên nhiều worker không được hỗ trợ.

## Chạy ứng dụng

Đặt các model GGUF theo `docker-compose.yaml`, rồi chạy:

LibreOffice là dependency bắt buộc để LiteParse xử lý các định dạng Office.

```bash
uv sync
docker compose up -d
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Ba dịch vụ model độc lập là LLM `:8080`, embedding `:8081`, và reranker `:8082`.
Có thể kiểm tra nhanh trước khi mở ứng dụng:

```bash
curl -fsS --max-time 2 http://127.0.0.1:8080/health
curl -fsS --max-time 2 http://127.0.0.1:8081/health
curl -fsS --max-time 2 http://127.0.0.1:8082/health
```

Mở <http://127.0.0.1:8000>. Các biến `LLM_URL`, `EMBED_URL`, `RERANK_URL`, và
`EMBEDDING_SIGNATURE` cho phép thay endpoint hoặc chủ động lên lịch reindex.

## Dữ liệu và API

`data/app.sqlite3` là nguồn dữ liệu có thẩm quyền: metadata session, document,
chunk, embedding và document job đều ở đây (WAL + foreign keys). File nguồn nằm
ở `data/uploads/<document-id>`; staging là `data/staging`.

Lần khởi động đầu tiên import không phá hủy `data/corpus/corpus.json` và
`data/history/chat_history.json`. Hai JSON này chỉ còn là bản sao phục hồi; runtime
không ghi chúng.

Session API:

- `POST/GET /api/sessions`, `PATCH/DELETE /api/sessions/{id}`
- `GET /api/sessions/{id}/messages`
- `POST /api/sessions/{id}/chat` (SSE), `POST /api/sessions/{id}/stop`

Document API:

- `POST/GET /api/documents`, `GET /api/documents/{id}/download`
- `POST /api/documents/{id}/retry`, `DELETE /api/documents/{id}`

Upload trả `202` ngay sau khi file, record `processing`, và job `ingest` đã được
commit. Worker chạy parser, overview và embedding ở nền; trạng thái document là
`processing`, `ready`, `failed`, hoặc `deleting`. Jobs bền vững là `queued`,
`running`, `succeeded`, `failed`, hoặc `cancelled`. Khởi động lại sẽ thu hồi job
đang chạy, dọn staging/file mồ côi, và rebuild snapshot từ embedding SQLite mà
không embed lại chunk sẵn sàng. Xóa tài liệu cũng là một job; tìm kiếm đã capture
snapshot cũ có thể hoàn tất, còn tìm kiếm mới không thấy tài liệu đã xóa.

Agent có một định nghĩa bất biến và chỉ hai tool chỉ-đọc: tìm chunk và lấy
overview. Mỗi run capture một snapshot; session SDK giữ history bền vững nhưng
model chỉ nhận tối đa 48 item thô, 12 message hoàn chỉnh, và 12.000 ký tự.

## Giới hạn và đồng thời

- Upload 25 MiB; parse tối đa 200 trang hoặc 300 giây; chat input 12.000 ký tự.
- Tool result và lỗi hiển thị bị chặn ở 48.000 và 500 ký tự tương ứng.
- Tối đa 4 LLM run khác session, 1 parser, 1 embedding request, 1 rerank request,
  và 2 RAG CPU threads. Một session chỉ có một run để chống double-click.

Parser chạy trong subprocess có thể hủy. BM25, vector preparation và xếp hạng
chạy ngoài event loop. Publication chỉ thay snapshot hoàn chỉnh sau commit SQLite,
nên document không thể `ready` trước khi searchable.

## Kiểm thử

```bash
uv run pytest -q
RUN_LIVE_MODEL_TESTS=1 uv run pytest -m live_model -q
RUN_PARSE_INTEGRATION=1 uv run pytest -m parse_integration -q
```

Hai lệnh sau là opt-in: chúng cần model service hoặc LiteParse/system converter
trên máy. Test thường dùng fake xác định và không cần Docker hay model.
