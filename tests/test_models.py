import pytest

from src.models import DataValidationError, DocumentRecord, JobRecord, Message, StoredChunk


@pytest.mark.parametrize(
    "record",
    [
        DocumentRecord("d", "file.pdf", "application/pdf", "ready", "", 0, "", 1.0, 2.0),
        StoredChunk("d", 0, ("p. 1",), "text", None, None),
        JobRecord("j", "d", "ingest", "queued", 0, 1.0, "", 1.0, None, None),
    ],
)
def test_durable_records_are_immutable(record: object) -> None:
    with pytest.raises((AttributeError, TypeError)):
        setattr(record, "id", "changed")


def test_parser_chunk_decoder_rejects_invalid_references() -> None:
    with pytest.raises(DataValidationError, match="chunk.refs"):
        StoredChunk("d", 0, (" ",), "text", None, None)


def test_visible_messages_are_limited_to_user_and_assistant() -> None:
    assert Message("user", "hello").content == "hello"
    with pytest.raises(DataValidationError, match="role"):
        Message("tool", "internal")  # type: ignore[arg-type]
