"""Document staging, disposable parser lifecycle, and atomic corpus commits."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
from typing import Any, Awaitable, TypeVar
from uuid import uuid4

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS, Settings
from src.llama import LlamaClient
from src.models import Chunk, Corpus, DataValidationError, Document
from src.rag import RagIndex


_T = TypeVar("_T")
_SAFE_CHAR = re.compile(r"[^\w .()-]+", re.UNICODE)


@dataclass(slots=True)
class LiveCorpus:
    """Mutable holder for the currently committed corpus snapshot."""

    value: Corpus


@dataclass(slots=True)
class RequestState:
    """Cancellation and process handles owned by one chat request."""

    request_id: str
    # The event reaches cooperative work; task/process handles stop hard work.
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[Any] | None = None
    parse_process: asyncio.subprocess.Process | None = None


class DocumentService:
    """Coordinate disposable parsing and transactional corpus updates."""

    def __init__(
        self,
        settings: Settings,
        llama: LlamaClient,
        live_corpus: LiveCorpus,
        rag: RagIndex,
    ) -> None:
        """Bind storage, model, corpus, and retrieval collaborators."""
        self._settings = settings
        self._llama = llama
        self._live = live_corpus
        self._rag = rag

    async def ingest(
        self,
        upload_name: str,
        content_or_upload: bytes | Any,
        request_state: RequestState,
    ) -> Document:
        """Parse and commit one upload, or leave no partial document."""
        safe_name = self._safe_name(upload_name)
        extension = Path(safe_name).suffix.lower()
        file_id = uuid4().hex
        staging = self._settings.staging_dir / request_state.request_id
        staged_input = staging / f"input{extension}"
        chunks_output = staging / "chunks.json"
        final_upload = self._upload_path(file_id, safe_name)

        if staging.exists():
            raise DataValidationError("request staging directory already exists")
        staging.mkdir(parents=True)
        try:
            content = await self._read_upload(content_or_upload)
            self._raise_if_cancelled(request_state)
            staged_input.write_bytes(content)

            process = await self._spawn_worker(
                [
                    sys.executable,
                    "-m",
                    "src.parse_worker",
                    "--input",
                    str(staged_input),
                    "--output",
                    str(chunks_output),
                    "--file-id",
                    file_id,
                    "--file-name",
                    safe_name,
                ]
            )
            request_state.parse_process = process
            try:
                return_code = await self._wait_for_worker(process, request_state)
            finally:
                if request_state.parse_process is process:
                    request_state.parse_process = None
            if return_code != 0:
                detail = await self._worker_error(process)
                raise RuntimeError(
                    f"document parser exited with code {return_code}"
                    + (f": {detail}" if detail else "")
                )

            chunks = self._load_worker_chunks(chunks_output, file_id, safe_name)
            self._raise_if_cancelled(request_state)
            overview = await self._await_or_cancel(
                self._create_overview(safe_name, chunks), request_state
            )
            document = Document(file_id, safe_name, overview, len(chunks))
            candidate_index = await self._await_or_cancel(
                self._rag.prepare_add(chunks), request_state
            )
            candidate_corpus = self._live.value.with_document(document, chunks)
            self._raise_if_cancelled(request_state)

            # Commit order keeps the live index behind durable file and corpus state.
            self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            os.replace(staged_input, final_upload)
            try:
                candidate_corpus.save(self._settings.corpus_path)
            except BaseException:
                final_upload.unlink(missing_ok=True)
                raise
            self._rag.install(candidate_index)
            self._live.value = candidate_corpus
            return document
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def delete(self, file_id: str) -> bool:
        """Transactionally remove one document from disk, corpus, and index."""
        document = next(
            (item for item in self._live.value.documents if item.file_id == file_id),
            None,
        )
        if document is None:
            return False
        candidate_corpus = self._live.value.without_document(file_id)
        candidate_index = self._rag.prepare_remove(file_id)
        upload = self._upload_path(document.file_id, document.file_name)
        temporary = self._settings.staging_dir / f"delete-{uuid4().hex}"
        moved = False
        temporary.parent.mkdir(parents=True, exist_ok=True)
        # Moving first allows the upload to be restored if corpus persistence fails.
        if upload.exists():
            os.replace(upload, temporary)
            moved = True
        try:
            candidate_corpus.save(self._settings.corpus_path)
        except BaseException:
            if moved:
                os.replace(temporary, upload)
            raise
        self._rag.install(candidate_index)
        self._live.value = candidate_corpus
        temporary.unlink(missing_ok=True)
        return True

    def clear(self) -> None:
        """Persist an empty corpus, then remove every committed upload."""
        previous = self._live.value
        empty = Corpus()
        candidate_index = self._rag.prepare_clear()
        empty.save(self._settings.corpus_path)
        self._rag.install(candidate_index)
        self._live.value = empty
        for document in previous.documents:
            self._upload_path(document.file_id, document.file_name).unlink(
                missing_ok=True
            )

    def prune_missing_uploads(self, corpus: Corpus) -> Corpus:
        """Reconcile persisted metadata with source files during startup."""
        self._settings.ensure_dirs()
        kept_documents = [
            document
            for document in corpus.documents
            if self._upload_path(document.file_id, document.file_name).is_file()
        ]
        kept_ids = {document.file_id for document in kept_documents}
        pruned = Corpus(
            kept_documents,
            [chunk for chunk in corpus.chunks if chunk.file_id in kept_ids],
        )
        referenced = {
            self._upload_path(document.file_id, document.file_name).resolve()
            for document in kept_documents
        }
        for path in self._settings.uploads_dir.iterdir():
            if path.is_file() and path.resolve() not in referenced:
                path.unlink(missing_ok=True)
        if pruned != corpus:
            pruned.save(self._settings.corpus_path)
        return pruned

    async def _spawn_worker(
        self, command: list[str]
    ) -> asyncio.subprocess.Process:
        """Spawn the parser in a new process group for bounded cleanup."""
        return await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

    async def _wait_for_worker(
        self,
        process: asyncio.subprocess.Process,
        state: RequestState,
    ) -> int:
        """Wait until the worker exits or request cancellation wins."""
        wait_task = asyncio.create_task(process.wait())
        cancel_task = asyncio.create_task(state.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {wait_task, cancel_task},
                timeout=self._settings.parse_timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await self._stop_worker_group(process)
                raise TimeoutError("document parser timed out")
            if cancel_task in done and state.cancel_event.is_set():
                await self._stop_worker_group(process)
                raise asyncio.CancelledError
            return await wait_task
        except asyncio.CancelledError:
            await self._stop_worker_group(process)
            raise
        finally:
            for task in (wait_task, cancel_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(wait_task, cancel_task, return_exceptions=True)

    async def _stop_worker_group(
        self, process: asyncio.subprocess.Process
    ) -> None:
        """Terminate the worker group, escalating to SIGKILL after grace."""
        if process.returncode is not None:
            await process.wait()
            return
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            await process.wait()
            return
        # Signal the group so OCR helpers cannot outlive their worker parent.
        with suppress(ProcessLookupError):
            os.killpg(process_group, signal.SIGTERM)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=self._settings.parse_termination_grace_seconds,
            )
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process_group, signal.SIGKILL)
            await process.wait()

    async def _await_or_cancel(
        self, awaitable: Awaitable[_T], state: RequestState
    ) -> _T:
        """Race asynchronous work against the request cancellation event."""
        work = asyncio.ensure_future(awaitable)
        cancellation = asyncio.create_task(state.cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {work, cancellation}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancellation in done and state.cancel_event.is_set():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                raise asyncio.CancelledError
            return await work
        except asyncio.CancelledError:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise
        finally:
            if not cancellation.done():
                cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)

    async def _read_upload(self, source: bytes | Any) -> bytes:
        """Read bytes or an upload stream without exceeding the size limit."""
        if isinstance(source, (bytes, bytearray, memoryview)):
            content = bytes(source)
        else:
            read = getattr(source, "read", None)
            if read is None:
                raise TypeError("upload content must be bytes or a readable upload")
            parts: list[bytes] = []
            size = 0
            while True:
                value = read(1024 * 1024)
                if inspect.isawaitable(value):
                    value = await value
                if not value:
                    break
                if not isinstance(value, bytes):
                    raise TypeError("upload reader must return bytes")
                size += len(value)
                if size > self._settings.max_upload_bytes:
                    raise DataValidationError("upload exceeds the size limit")
                parts.append(value)
            content = b"".join(parts)
        if not content:
            raise DataValidationError("upload is empty")
        if len(content) > self._settings.max_upload_bytes:
            raise DataValidationError("upload exceeds the size limit")
        return content

    async def _create_overview(
        self, file_name: str, chunks: list[Chunk]
    ) -> str:
        """Generate a bounded overview from parsed chunks."""
        sections = [
            f"[{', '.join(chunk.refs)}]\n{chunk.text}" for chunk in chunks
        ]
        context = "\n\n---\n\n".join(sections)[: self._settings.max_context_chars]
        overview = await self._llama.complete_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Tạo overview tiếng Việt ngắn gọn cho tài liệu: tóm tắt, "
                        "dàn ý và các điểm chính, tối đa 300 từ. "
                        "Chỉ dùng nội dung được cung cấp."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Tài liệu {file_name}:\n\n{context}",
                },
            ],
            max_tokens=768,
            temperature=0.1,
        )
        if not overview.strip():
            raise DataValidationError("overview model returned empty content")
        return overview.strip()

    @staticmethod
    async def _worker_error(process: asyncio.subprocess.Process) -> str:
        """Read a bounded, single-line parser error message."""
        if process.stderr is None:
            return ""
        value = await process.stderr.read(2_048)
        return " ".join(value.decode("utf-8", errors="replace").split())[:500]

    @staticmethod
    def _load_worker_chunks(
        path: Path, file_id: str, file_name: str
    ) -> list[Chunk]:
        """Load worker output and verify its upload identity and ordering."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            raw_chunks = value["chunks"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DataValidationError("parser produced invalid chunks JSON") from exc
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise DataValidationError("parser produced no chunks")
        chunks = [Chunk.from_dict(item) for item in raw_chunks]
        if any(
            chunk.file_id != file_id or chunk.file_name != file_name
            for chunk in chunks
        ):
            raise DataValidationError("parser chunk metadata does not match upload")
        if [chunk.chunk_id for chunk in chunks] != list(range(len(chunks))):
            raise DataValidationError("parser chunk IDs are not sequential")
        return chunks

    @staticmethod
    def _safe_name(upload_name: str) -> str:
        """Normalize an upload display name and enforce supported suffixes."""
        if not isinstance(upload_name, str) or "\x00" in upload_name:
            raise DataValidationError("upload filename is invalid")
        basename = Path(upload_name.replace("\\", "/")).name.strip()
        basename = _SAFE_CHAR.sub("_", basename).strip(" .")
        if not basename or basename in {".", ".."}:
            raise DataValidationError("upload filename is empty")
        if len(basename) > 180:
            raise DataValidationError("upload filename is too long")
        if Path(basename).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
            raise DataValidationError(
                f"định dạng file không được hỗ trợ; định dạng hợp lệ: {supported}"
            )
        return basename

    def _upload_path(self, file_id: str, file_name: str) -> Path:
        """Return the committed source path for a document."""
        return self._settings.uploads_dir / f"{file_id}_{file_name}"

    @staticmethod
    def _raise_if_cancelled(state: RequestState) -> None:
        """Stop a transaction before its next irreversible step."""
        if state.cancel_event.is_set():
            raise asyncio.CancelledError
