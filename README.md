# Local RAG Chatbot

Pet project RAG chạy trên một máy Linux có GPU NVIDIA. Ứng dụng giữ kiến trúc nhỏ:

- một FastAPI process chứa API, history, corpus và hybrid RAG index;
- ba llama.cpp CUDA container cho LLM, embedding và reranking;
- một LiteParse worker process group tạm thời cho mỗi lần upload, tự thoát sau khi tạo chunk JSON.

Không có parser daemon, task queue, vector database, agent framework, Torch hay `llama-cpp-python` trong application process.

## Yêu cầu

- Python 3.12 và [uv](https://docs.astral.sh/uv/);
- Docker Compose cùng NVIDIA Container Toolkit;
- LibreOffice cho DOCX;
- Tesseract với language data `vie` và `eng`;
- các model sau trong `models/`:
  - `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`;
  - `mtp-gemma-4-E4B-it.gguf`;
  - `bge-m3-Q8_0.gguf`;
  - `bge-reranker-v2-m3-Q8_0.gguf`.

## Cài đặt

```bash
uv sync --group dev
libreoffice --headless --version
tesseract --list-langs
uv run python -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('BAAI/bge-m3')"
docker compose up -d
```

Tokenizer BGE-M3 cần được tải trước một lần. Parse worker dùng cache local và không âm thầm chuyển sang character counting nếu cache thiếu.

Kiểm tra model server:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8081/health
curl -fsS http://127.0.0.1:8082/health
```

## Chạy ứng dụng

```bash
uv run python -m src.main
```

Hoặc:

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Chỉ chạy một Uvicorn worker vì corpus, RAG index và active-request slot nằm trong memory. Mở <http://127.0.0.1:8000> để dùng giao diện.

PDF/DOCX được parse với LiteParse, selective OCR `vie+eng`, sau đó Markdown được chia tối đa 1024 BGE-M3 token. Chat bị khóa trong suốt ingest và agent pipeline. Stop sẽ hủy phần chưa commit; tài liệu đã atomic-commit vẫn được giữ lại.

## Kiểm thử

```bash
uv run pytest -v
```

Hai nhóm kiểm thử nặng được bật thủ công:

```bash
RUN_PARSE_INTEGRATION=1 uv run pytest tests/test_parse_worker.py -m parse_integration -v
RUN_LIVE_MODEL_TEST=1 uv run pytest tests/test_agent_eval.py -m live_model -v -s
```

Parse integration dùng trực tiếp `docs/test.pdf` và `docs/DACSN.docx`; live-model evaluation cần cả ba container đang healthy.
