"""Isolated, cancellable parser subprocess lifecycle."""

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import tempfile

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS, Settings
from src.models import Chunk, DataValidationError


_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKDOWN_REFERENCE = re.compile(r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$")
_TAG = re.compile(r"<!--[\s\S]*?-->|<[^>]*>")


def validate_parsed_chunks(chunks: list[Chunk]) -> None:
    """Reject syntactically valid parser output without meaningful content."""
    if not chunks:
        raise DataValidationError("parser produced no chunks")
    text = "\n".join(chunk.text for chunk in chunks)
    text = _MARKDOWN_REFERENCE.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _TAG.sub("", text)
    if sum(character.isalnum() for character in text) < 8:
        raise DataValidationError("parser produced no meaningful content")


class ParserService:
    """Run one disposable LiteParse worker at a time and reap its process group."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.parser_concurrency)
        self._active_lock = asyncio.Lock()
        self._active_process: asyncio.subprocess.Process | None = None

    async def parse(
        self, document_id: str, file_name: str, source_path: Path
    ) -> list[Chunk]:
        """Parse one committed source after restoring its validated suffix in staging."""
        self._validate_input(document_id, file_name, source_path)
        suffix = Path(file_name).suffix
        self._settings.staging_dir.mkdir(parents=True, exist_ok=True)
        async with self._semaphore:
            with tempfile.TemporaryDirectory(
                dir=self._settings.staging_dir, prefix="parse-"
            ) as directory:
                staging = Path(directory)
                staged_source = staging / f"input{suffix}"
                try:
                    os.link(source_path, staged_source)
                except OSError:
                    shutil.copy2(source_path, staged_source)
                output_path = staging / "chunks.json"
                process = await self._spawn_worker(
                    [
                        sys.executable,
                        "-m",
                        "src.parse_worker",
                        "--input",
                        str(staged_source),
                        "--output",
                        str(output_path),
                        "--file-id",
                        document_id,
                        "--file-name",
                        file_name,
                    ]
                )
                await self._set_active(process)
                try:
                    return_code = await self._wait_for_worker(process)
                    if return_code != 0:
                        detail = await self._worker_error(process)
                        raise RuntimeError(
                            f"document parser exited with code {return_code}"
                            + (f": {detail}" if detail else "")
                        )
                    chunks = self._load_worker_chunks(
                        output_path, document_id, file_name
                    )
                    validate_parsed_chunks(chunks)
                    return chunks
                except asyncio.CancelledError:
                    await self._stop_worker_group(process)
                    raise
                finally:
                    await self._clear_active(process)

    async def cancel_active(self) -> None:
        """Terminate and reap the current parser group, if a parse is running."""
        async with self._active_lock:
            process = self._active_process
        if process is not None:
            await self._stop_worker_group(process)

    @property
    def active_process(self) -> asyncio.subprocess.Process | None:
        """Expose the current process handle for legacy request-state compatibility."""
        return self._active_process

    async def _set_active(self, process: asyncio.subprocess.Process) -> None:
        async with self._active_lock:
            self._active_process = process

    async def _clear_active(self, process: asyncio.subprocess.Process) -> None:
        async with self._active_lock:
            if self._active_process is process:
                self._active_process = None

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

    async def _wait_for_worker(self, process: asyncio.subprocess.Process) -> int:
        """Wait under the configured timeout and terminate the full group on expiry."""
        try:
            return await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=self._settings.parse_timeout_seconds,
            )
        except TimeoutError:
            await self._stop_worker_group(process)
            raise TimeoutError("document parser timed out")

    async def _stop_worker_group(
        self, process: asyncio.subprocess.Process
    ) -> None:
        """Terminate a worker group, escalating after the configured grace period."""
        if process.returncode is not None:
            await process.wait()
            return
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            await process.wait()
            return
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

    @staticmethod
    async def _worker_error(process: asyncio.subprocess.Process) -> str:
        """Read a bounded, single-line worker error suitable for an application log."""
        if process.stderr is None:
            return ""
        value = await process.stderr.read(2_048)
        return " ".join(value.decode("utf-8", errors="replace").split())[:500]

    @staticmethod
    def _load_worker_chunks(
        path: Path, document_id: str, file_name: str
    ) -> list[Chunk]:
        """Decode worker output and enforce the requested document identity/order."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            raw_chunks = value["chunks"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise DataValidationError("parser produced invalid chunks JSON") from exc
        if not isinstance(raw_chunks, list) or not raw_chunks:
            raise DataValidationError("parser produced no chunks")
        chunks = [Chunk.from_dict(item) for item in raw_chunks]
        if any(
            chunk.file_id != document_id or chunk.file_name != file_name
            for chunk in chunks
        ):
            raise DataValidationError("parser chunk metadata does not match upload")
        if [chunk.chunk_id for chunk in chunks] != list(range(len(chunks))):
            raise DataValidationError("parser chunk IDs are not sequential")
        return chunks

    @staticmethod
    def _validate_input(document_id: str, file_name: str, source_path: Path) -> None:
        """Validate the trusted metadata needed to reconstruct the parser input name."""
        if not isinstance(document_id, str) or not document_id:
            raise DataValidationError("document ID is invalid")
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or Path(file_name).suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS
        ):
            raise DataValidationError("display filename is unsafe or unsupported")
        if not source_path.is_file():
            raise DataValidationError("document source file is missing")
