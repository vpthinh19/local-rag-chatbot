"""Small application configuration with one replaceable data root."""

from dataclasses import dataclass, field
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    max_context_chars: int = 48_000
    embedding_batch_size: int = 32
    lexical_candidate_limit: int = 24
    semantic_candidate_limit: int = 24
    fused_candidate_limit: int = 16
    final_chunk_limit: int = 5
    parse_termination_grace_seconds: float = 3.0
    max_parse_pages: int = 200
    tokenizer_name: str = "BAAI/bge-m3"

    def __post_init__(self) -> None:
        """Normalize paths and reject nonpositive resource limits."""
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        numeric_limits = {
            "http_connect_timeout": self.http_connect_timeout,
            "http_read_timeout": self.http_read_timeout,
            "http_write_timeout": self.http_write_timeout,
            "http_pool_timeout": self.http_pool_timeout,
            "max_upload_bytes": self.max_upload_bytes,
            "max_context_chars": self.max_context_chars,
            "embedding_batch_size": self.embedding_batch_size,
            "lexical_candidate_limit": self.lexical_candidate_limit,
            "semantic_candidate_limit": self.semantic_candidate_limit,
            "fused_candidate_limit": self.fused_candidate_limit,
            "final_chunk_limit": self.final_chunk_limit,
            "parse_termination_grace_seconds": self.parse_termination_grace_seconds,
            "max_parse_pages": self.max_parse_pages,
        }
        invalid = [name for name, value in numeric_limits.items() if value <= 0]
        if invalid:
            raise ValueError(f"settings must be positive: {', '.join(invalid)}")

    @property
    def uploads_dir(self) -> Path:
        """Return the directory containing committed source files."""
        return self.data_dir / "uploads"

    @property
    def staging_dir(self) -> Path:
        """Return the directory for request-scoped temporary files."""
        return self.data_dir / "staging"

    @property
    def corpus_path(self) -> Path:
        """Return the persisted corpus manifest path."""
        return self.data_dir / "corpus" / "corpus.json"

    @property
    def history_path(self) -> Path:
        """Return the persisted chat history path."""
        return self.data_dir / "history" / "chat_history.json"

    def ensure_dirs(self) -> None:
        """Create every application-owned data directory."""
        for path in (
            self.uploads_dir,
            self.staging_dir,
            self.corpus_path.parent,
            self.history_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
