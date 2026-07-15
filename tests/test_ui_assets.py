from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "templates" / "index.html"
SCRIPT = ROOT / "src" / "static" / "script.js"
STYLE = ROOT / "src" / "static" / "style.css"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)


def test_existing_ui_assets_are_present() -> None:
    assert TEMPLATE.is_file()
    assert SCRIPT.is_file()
    assert STYLE.is_file()


def test_existing_control_ids_are_preserved() -> None:
    collector = _IdCollector()
    collector.feed(TEMPLATE.read_text(encoding="utf-8"))

    assert {
        "sidebar",
        "documents-list",
        "toggle-sidebar-btn",
        "prompt-form",
        "prompt-input",
        "file-input",
        "cancel-file-btn",
        "add-file-btn",
        "stop-response-btn",
        "send-prompt-btn",
        "theme-toggle-btn",
        "delete-chats-btn",
    } <= collector.ids


def test_existing_api_routes_remain_wired() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    for route in (
        "/api/chat",
        "/api/stop",
        "/api/chat-history",
        "/api/clear-chat",
        "/api/documents",
    ):
        assert route in script


def test_saved_and_submitted_messages_use_text_content() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'textContent = msg.content' in script
    assert 'textContent = userMessage' in script
    assert 'textContent = fullResponse' in script


def test_filenames_are_not_interpolated_into_inner_html() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert '${doc.file_name}' not in script
    assert '${uploadedFile.name}' not in script
