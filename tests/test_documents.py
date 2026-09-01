from pathlib import Path

import pytest

from src.config import Settings
from src.database import Database
from src.documents import DocumentService
from src.models import DataValidationError


class Upload:
    def __init__(self, name: str, content: bytes, media_type: str = "application/pdf") -> None:
        self.filename = name
        self.content_type = media_type
        self._content = content
        self._offset = 0

    async def read(self, size: int) -> bytes:
        result = self._content[self._offset : self._offset + size]
        self._offset += len(result)
        return result


@pytest.mark.asyncio
async def test_upload_commits_a_processing_document_and_durable_job(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    documents = DocumentService(settings, database)
    document = await documents.create_upload(Upload("report.pdf", b"pdf"))
    assert document.status == "processing"
    assert documents.source_path(document.id).read_bytes() == b"pdf"
    assert await database.read(lambda conn: conn.execute("SELECT operation, state FROM document_jobs WHERE document_id = ?", (document.id,)).fetchone()) == ("ingest", "queued")


@pytest.mark.asyncio
async def test_upload_rejects_untrusted_metadata_before_committing(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    database = Database(settings.database_path, 2_000)
    await database.initialize()
    with pytest.raises(DataValidationError):
        await DocumentService(settings, database).create_upload(Upload("bad.txt", b"x", "text/plain"))
