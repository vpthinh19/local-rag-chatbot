"""Small application configuration with one replaceable data root."""

from dataclasses import dataclass, field
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# LiteParse converts office files with LibreOffice and handles images natively.
PDF_EXTENSIONS = frozenset({".pdf"})
OFFICE_EXTENSIONS = frozenset(
    {
        ".csv",
        ".doc",
        ".docm",
        ".docx",
        ".dot",
        ".dotm",
        ".dotx",
        ".key",
        ".numbers",
        ".odp",
        ".ods",
        ".odt",
        ".otp",
        ".ots",
        ".ott",
        ".pages",
        ".pot",
        ".potm",
        ".potx",
        ".ppt",
        ".pptm",
        ".pptx",
        ".rtf",
        ".tsv",
        ".xls",
        ".xlsb",
        ".xlsm",
        ".xlsx",
    }
)
IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".tif", ".tiff", ".webp"}
)
TEXT_EXTENSIONS = frozenset({".log", ".markdown", ".md", ".txt"})
SUPPORTED_DOCUMENT_EXTENSIONS = (
    PDF_EXTENSIONS | OFFICE_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS
)


def _url_from_env(name: str, default: str) -> str:
    """Read a service URL without a trailing slash."""
    return os.getenv(name, default).rstrip("/")


@dataclass(frozen=True, slots=True)
class Settings:
    """Centralize filesystem, model endpoint, and resource limits."""

    data_dir: Path = PROJECT_ROOT / "data"

    # Independent llama.cpp-compatible model services.
    llm_url: str = field(
        default_factory=lambda: _url_from_env("LLM_URL", "http://127.0.0.1:8080")
    )
    embed_url: str = field(
        default_factory=lambda: _url_from_env("EMBED_URL", "http://127.0.0.1:8081")
    )
    rerank_url: str = field(
        default_factory=lambda: _url_from_env("RERANK_URL", "http://127.0.0.1:8082")
    )

    # Shared HTTP bounds prevent a stalled model from blocking the only request slot.
    http_connect_timeout: float = 5.0
    http_read_timeout: float = 120.0
    http_write_timeout: float = 30.0
    http_pool_timeout: float = 5.0

    # Ingestion and retrieval budgets keep memory and prompt size predictable.
    max_upload_bytes: int = 25 * 1024 * 1024
    max_message_chars: int = 12_000
    max_context_chars: int = 48_000
    embedding_batch_size: int = 32
    lexical_candidate_limit: int = 24
    semantic_candidate_limit: int = 24
    fused_candidate_limit: int = 16
    final_chunk_limit: int = 5
    parse_termination_grace_seconds: float = 3.0
    parse_timeout_seconds: float = 300.0
    max_parse_pages: int = 200
    tokenizer_name: str = "BAAI/bge-m3"
    embedding_signature: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_SIGNATURE", "BAAI/bge-m3")
    )

    # Durable agent, session, and job limits.
    agent_model: str = "local"
    agent_max_turns: int = 4
    session_raw_item_limit: int = 48
    session_visible_message_limit: int = 12
    session_context_chars: int = 12_000
    session_title_chars: int = 80
    job_max_attempts: int = 3
    job_retry_base_seconds: float = 1.0
    database_busy_timeout_ms: int = 5_000

    # Per-resource gates match the measured local service capacities.
    llm_concurrency: int = 4
    parser_concurrency: int = 1
    embedding_concurrency: int = 1
    rerank_concurrency: int = 1
    rag_cpu_workers: int = 2

    def __post_init__(self) -> None:
        """Normalize paths and reject nonpositive resource limits."""
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        numeric_limits = {
            "http_connect_timeout": self.http_connect_timeout,
            "http_read_timeout": self.http_read_timeout,
            "http_write_timeout": self.http_write_timeout,
            "http_pool_timeout": self.http_pool_timeout,
            "max_upload_bytes": self.max_upload_bytes,
            "max_message_chars": self.max_message_chars,
            "max_context_chars": self.max_context_chars,
            "embedding_batch_size": self.embedding_batch_size,
            "lexical_candidate_limit": self.lexical_candidate_limit,
            "semantic_candidate_limit": self.semantic_candidate_limit,
            "fused_candidate_limit": self.fused_candidate_limit,
            "final_chunk_limit": self.final_chunk_limit,
            "parse_termination_grace_seconds": self.parse_termination_grace_seconds,
            "parse_timeout_seconds": self.parse_timeout_seconds,
            "max_parse_pages": self.max_parse_pages,
            "agent_max_turns": self.agent_max_turns,
            "session_raw_item_limit": self.session_raw_item_limit,
            "session_visible_message_limit": self.session_visible_message_limit,
            "session_context_chars": self.session_context_chars,
            "session_title_chars": self.session_title_chars,
            "job_max_attempts": self.job_max_attempts,
            "job_retry_base_seconds": self.job_retry_base_seconds,
            "database_busy_timeout_ms": self.database_busy_timeout_ms,
            "llm_concurrency": self.llm_concurrency,
            "parser_concurrency": self.parser_concurrency,
            "embedding_concurrency": self.embedding_concurrency,
            "rerank_concurrency": self.rerank_concurrency,
            "rag_cpu_workers": self.rag_cpu_workers,
        }
        invalid = [name for name, value in numeric_limits.items() if value <= 0]
        if invalid:
            raise ValueError(f"settings must be positive: {', '.join(invalid)}")
        if not self.embedding_signature.strip():
            raise ValueError("embedding_signature must be nonempty")

    @property
    def uploads_dir(self) -> Path:
        """Return the directory containing committed source files."""
        return self.data_dir / "uploads"

    @property
    def staging_dir(self) -> Path:
        """Return the directory for request-scoped temporary files."""
        return self.data_dir / "staging"

    @property
    def database_path(self) -> Path:
        """Return the application-owned SQLite database path."""
        return self.data_dir / "app.sqlite3"

    @property
    def legacy_corpus_path(self) -> Path:
        """Return the read-only legacy corpus backup used by migration."""
        return self.data_dir / "corpus" / "corpus.json"

    @property
    def legacy_history_path(self) -> Path:
        """Return the read-only legacy history backup used by migration."""
        return self.data_dir / "history" / "chat_history.json"

    def ensure_dirs(self) -> None:
        """Create every application-owned data directory."""
        for path in (
            self.uploads_dir,
            self.staging_dir,
            self.legacy_corpus_path.parent,
            self.legacy_history_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
