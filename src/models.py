"""Small validated records at application and parser boundaries."""

from dataclasses import dataclass
import math
from typing import Literal, Mapping


class DataValidationError(ValueError):
    """Persisted or external data does not match the public DTO contract."""


DocumentStatus = Literal["processing", "ready", "failed", "deleting"]
JobOperation = Literal["ingest", "reindex", "delete"]
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


def _timestamp(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{label} must be a finite number")
    return result


def _string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a nonempty string"
        raise DataValidationError(f"{label} must be {suffix}")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataValidationError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    file_name: str
    media_type: str
    status: DocumentStatus
    overview: str
    chunk_count: int
    error: str
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        _string(self.id, "document.id")
        _string(self.file_name, "document.file_name")
        _string(self.media_type, "document.media_type")
        if self.status not in {"processing", "ready", "failed", "deleting"}:
            raise DataValidationError("document.status is invalid")
        _string(self.overview, "document.overview", allow_empty=True)
        _integer(self.chunk_count, "document.chunk_count")
        _string(self.error, "document.error", allow_empty=True)
        _timestamp(self.created_at, "document.created_at")
        _timestamp(self.updated_at, "document.updated_at")


@dataclass(frozen=True, slots=True)
class StoredChunk:
    document_id: str
    chunk_id: int
    refs: tuple[str, ...]
    text: str
    embedding: bytes | None
    embedding_dim: int | None

    def __post_init__(self) -> None:
        _string(self.document_id, "chunk.document_id")
        _integer(self.chunk_id, "chunk.chunk_id")
        if not isinstance(self.refs, tuple) or not all(
            isinstance(ref, str) and ref.strip() for ref in self.refs
        ):
            raise DataValidationError("chunk.refs must contain nonempty strings")
        _string(self.text, "chunk.text")
        if self.embedding is not None and not isinstance(self.embedding, bytes):
            raise DataValidationError("chunk.embedding must be bytes or None")
        if self.embedding_dim is not None:
            _integer(self.embedding_dim, "chunk.embedding_dim")
            if self.embedding_dim == 0:
                raise DataValidationError("chunk.embedding_dim must be positive")
        if (self.embedding is None) != (self.embedding_dim is None):
            raise DataValidationError("chunk.embedding and dimension must agree")


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    document_id: str
    operation: JobOperation
    state: JobState
    attempts: int
    next_attempt_at: float
    error: str
    created_at: float
    started_at: float | None
    finished_at: float | None

    def __post_init__(self) -> None:
        _string(self.id, "job.id")
        _string(self.document_id, "job.document_id")
        if self.operation not in {"ingest", "reindex", "delete"}:
            raise DataValidationError("job.operation is invalid")
        if self.state not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise DataValidationError("job.state is invalid")
        _integer(self.attempts, "job.attempts")
        _timestamp(self.next_attempt_at, "job.next_attempt_at")
        _string(self.error, "job.error", allow_empty=True)
        _timestamp(self.created_at, "job.created_at")
        if self.started_at is not None:
            _timestamp(self.started_at, "job.started_at")
        if self.finished_at is not None:
            _timestamp(self.finished_at, "job.finished_at")


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    title: str
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        _string(self.id, "session.id")
        _string(self.title, "session.title")
        _timestamp(self.created_at, "session.created_at")
        _timestamp(self.updated_at, "session.updated_at")


@dataclass(frozen=True, slots=True)
class MigrationReport:
    imported_documents: int = 0
    imported_messages: int = 0
    reindexed_documents: int = 0
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.imported_documents, "migration.imported_documents")
        _integer(self.imported_messages, "migration.imported_messages")
        _integer(self.reindexed_documents, "migration.reindexed_documents")
        if not isinstance(self.errors, tuple) or not all(
            isinstance(error, str) and error and len(error) <= 500
            for error in self.errors
        ):
            raise DataValidationError("migration.errors must contain bounded messages")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A parser-produced retrievable span with source-page references."""

    file_id: str
    file_name: str
    chunk_id: int
    refs: list[str]
    text: str

    def __post_init__(self) -> None:
        _string(self.file_id, "chunk.file_id")
        _string(self.file_name, "chunk.file_name")
        _integer(self.chunk_id, "chunk.chunk_id")
        if not isinstance(self.refs, list) or not all(
            isinstance(ref, str) and ref.strip() for ref in self.refs
        ):
            raise DataValidationError("chunk.refs must contain nonempty strings")
        _string(self.text, "chunk.text")
        object.__setattr__(self, "refs", list(self.refs))

    def to_dict(self) -> dict[str, object]:
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "chunk_id": self.chunk_id,
            "refs": list(self.refs),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, value: object) -> "Chunk":
        if not isinstance(value, Mapping):
            raise DataValidationError("chunk must be a JSON object")
        try:
            refs = value.get("refs", [])
            if not isinstance(refs, list):
                raise DataValidationError("chunk.refs must be a JSON array")
            return cls(
                _string(value["file_id"], "chunk.file_id"),
                _string(value["file_name"], "chunk.file_name"),
                _integer(value["chunk_id"], "chunk.chunk_id"),
                [_string(ref, "chunk.refs[]") for ref in refs],
                _string(value["text"], "chunk.text"),
            )
        except KeyError as exc:
            raise DataValidationError(f"chunk is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise DataValidationError("message.role must be user or assistant")
        _string(self.content, "message.content")
