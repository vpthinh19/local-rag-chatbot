"""Validated data transfer objects and atomic JSON persistence."""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping


class DataValidationError(ValueError):
    """Persisted or external data does not match the public DTO contract."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"{label} must be a JSON object")
    return value


def _string(
    value: object, label: str, *, allow_empty: bool = False
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "a string" if allow_empty else "a nonempty string"
        raise DataValidationError(f"{label} must be {suffix}")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataValidationError(f"{label} must be a nonnegative integer")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise DataValidationError(f"{label} must be a JSON array")
    return value


def _read_json(path: Path, label: str) -> object | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid {label} JSON: {exc.msg}") from exc


def _atomic_json_save(path: Path, value: object) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class Chunk:
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
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "Chunk":
        data = _mapping(value, "chunk")
        try:
            refs = _list(data.get("refs", []), "chunk.refs")
            return cls(
                _string(data["file_id"], "chunk.file_id"),
                _string(data["file_name"], "chunk.file_name"),
                _integer(data["chunk_id"], "chunk.chunk_id"),
                [_string(ref, "chunk.refs[]") for ref in refs],
                _string(data["text"], "chunk.text"),
            )
        except KeyError as exc:
            raise DataValidationError(f"chunk is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class Document:
    file_id: str
    file_name: str
    overview: str
    chunk_count: int

    def __post_init__(self) -> None:
        _string(self.file_id, "document.file_id")
        _string(self.file_name, "document.file_name")
        _string(self.overview, "document.overview", allow_empty=True)
        _integer(self.chunk_count, "document.chunk_count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "Document":
        data = _mapping(value, "document")
        try:
            overview = data.get("overview", data.get("summary", ""))
            return cls(
                _string(data["file_id"], "document.file_id"),
                _string(data["file_name"], "document.file_name"),
                _string(overview, "document.overview", allow_empty=True),
                _integer(data["chunk_count"], "document.chunk_count"),
            )
        except KeyError as exc:
            raise DataValidationError(f"document is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"user", "assistant"}:
            raise DataValidationError("message.role must be user or assistant")
        _string(self.content, "message.content")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "Message":
        data = _mapping(value, "message")
        try:
            role = _string(data["role"], "message.role")
            if role not in {"user", "assistant"}:
                raise DataValidationError("message.role must be user or assistant")
            return cls(role, _string(data["content"], "message.content"))
        except KeyError as exc:
            raise DataValidationError(f"message is missing {exc.args[0]}") from exc


@dataclass(frozen=True, slots=True)
class Corpus:
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.documents, list) or not all(
            isinstance(document, Document) for document in self.documents
        ):
            raise DataValidationError("corpus.documents must contain Document values")
        if not isinstance(self.chunks, list) or not all(
            isinstance(chunk, Chunk) for chunk in self.chunks
        ):
            raise DataValidationError("corpus.chunks must contain Chunk values")

        document_by_id: dict[str, Document] = {}
        for document in self.documents:
            if document.file_id in document_by_id:
                raise DataValidationError(
                    f"duplicate document file_id: {document.file_id}"
                )
            document_by_id[document.file_id] = document

        counts = {file_id: 0 for file_id in document_by_id}
        chunk_ids: set[tuple[str, int]] = set()
        for chunk in self.chunks:
            document = document_by_id.get(chunk.file_id)
            if document is None:
                raise DataValidationError(
                    f"chunk references unknown document: {chunk.file_id}"
                )
            if chunk.file_name != document.file_name:
                raise DataValidationError(
                    f"chunk filename mismatch for document: {chunk.file_id}"
                )
            key = (chunk.file_id, chunk.chunk_id)
            if key in chunk_ids:
                raise DataValidationError(
                    f"duplicate chunk_id {chunk.chunk_id} for {chunk.file_id}"
                )
            chunk_ids.add(key)
            counts[chunk.file_id] += 1

        for document in self.documents:
            if document.chunk_count != counts[document.file_id]:
                raise DataValidationError(
                    f"document chunk_count mismatch for {document.file_id}"
                )

        object.__setattr__(self, "documents", list(self.documents))
        object.__setattr__(self, "chunks", list(self.chunks))

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": [document.to_dict() for document in self.documents],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, value: object) -> "Corpus":
        data = _mapping(value, "corpus")
        raw_documents = data.get("documents", data.get("summaries", []))
        raw_chunks = data.get("chunks", [])
        documents = _list(raw_documents, "corpus.documents")
        chunks = _list(raw_chunks, "corpus.chunks")
        return cls(
            [Document.from_dict(document) for document in documents],
            [Chunk.from_dict(chunk) for chunk in chunks],
        )

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        value = _read_json(Path(path), "corpus")
        return cls() if value is None else cls.from_dict(value)

    def save(self, path: Path) -> None:
        _atomic_json_save(Path(path), self.to_dict())

    def with_document(
        self, document: Document, chunks: list[Chunk]
    ) -> "Corpus":
        if document.file_id in {item.file_id for item in self.documents}:
            raise DataValidationError(
                f"duplicate document file_id: {document.file_id}"
            )
        return Corpus(self.documents + [document], self.chunks + list(chunks))

    def without_document(self, file_id: str) -> "Corpus":
        return Corpus(
            [document for document in self.documents if document.file_id != file_id],
            [chunk for chunk in self.chunks if chunk.file_id != file_id],
        )


@dataclass(frozen=True, slots=True)
class History:
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, list) or not all(
            isinstance(message, Message) for message in self.messages
        ):
            raise DataValidationError("history.messages must contain Message values")
        object.__setattr__(self, "messages", list(self.messages))

    def to_dict(self) -> dict[str, object]:
        return {"messages": [message.to_dict() for message in self.messages]}

    @classmethod
    def from_dict(cls, value: object) -> "History":
        data = _mapping(value, "history")
        raw_messages = _list(data.get("messages", []), "history.messages")
        messages: list[Message] = []
        for value in raw_messages:
            item = _mapping(value, "message")
            if item.get("role") in {"user", "assistant"}:
                messages.append(
                    Message.from_dict(
                        {"role": item.get("role"), "content": item.get("content")}
                    )
                )
        return cls(messages)

    @classmethod
    def load(cls, path: Path) -> "History":
        value = _read_json(Path(path), "history")
        return cls() if value is None else cls.from_dict(value)

    def save(self, path: Path) -> None:
        _atomic_json_save(Path(path), self.to_dict())

    def with_turn(self, user: str, assistant: str) -> "History":
        return History(
            self.messages + [Message("user", user), Message("assistant", assistant)]
        )
