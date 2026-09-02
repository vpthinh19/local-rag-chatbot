"""Static UI contracts and pure browser-state behavior."""

from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

from src.config import SUPPORTED_DOCUMENT_EXTENSIONS

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "templates" / "index.html"
SCRIPT = ROOT / "src" / "static" / "script.js"
STATE = ROOT / "src" / "static" / "state.mjs"
MARKDOWN = ROOT / "src" / "static" / "markdown.mjs"
STYLE = ROOT / "src" / "static" / "style.css"


class TemplateTree(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.elements: dict[str, tuple[str, dict[str, str], tuple[str, ...]]] = {}
        self._parents: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        identifier = values.get("id")
        if identifier:
            self.elements[identifier] = tag, values, tuple(self._parents)
        if tag not in self.VOID_TAGS:
            self._parents.append(identifier or "")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID_TAGS:
            return
        if self._parents:
            self._parents.pop()

    def has_button(self, identifier: str) -> bool:
        return self.elements.get(identifier, ("", {}, ()))[0] == "button"

    def has_list(self, identifier: str) -> bool:
        return self.elements.get(identifier, ("", {}, ()))[0] in {"ul", "ol"}

    def has_input(self, identifier: str) -> bool:
        return self.elements.get(identifier, ("", {}, ()))[0] == "input"

    def is_descendant(self, child: str, parent: str) -> bool:
        return parent in self.elements[child][2]


def parse_template() -> TemplateTree:
    tree = TemplateTree()
    tree.feed(TEMPLATE.read_text(encoding="utf-8"))
    return tree


def run_state_function(name: str, value: object) -> object:
    program = (
        f"import {{ {name} }} from {json.dumps(STATE.as_uri())};"
        f"const value = JSON.parse({json.dumps(json.dumps(value))});"
        f"console.log(JSON.stringify({name}(value)));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_ui_assets_are_present() -> None:
    assert all(path.is_file() for path in (TEMPLATE, SCRIPT, STATE, STYLE))


def test_template_separates_upload_from_prompt_form() -> None:
    tree = parse_template()
    assert tree.has_button("new-session-btn")
    assert tree.has_list("sessions-list")
    assert tree.has_input("document-file-input")
    assert not tree.is_descendant("document-file-input", "prompt-form")
    assert tree.has_button("upload-document-btn")
    assert tree.has_button("stop-response-btn")


def test_template_has_two_vertical_management_sidebars_and_balanced_controls() -> None:
    tree = parse_template()
    assert tree.is_descendant("new-session-btn", "session-sidebar")
    assert tree.is_descendant("sessions-list", "session-sidebar")
    assert tree.is_descendant("upload-document-btn", "document-sidebar")
    assert tree.is_descendant("documents-list", "document-sidebar")
    for identifier in ("toggle-session-sidebar-btn", "toggle-document-sidebar-btn"):
        assert tree.has_button(identifier)
        assert tree.is_descendant(identifier, "top-bar")
    assert tree.has_button("theme-toggle-btn")
    assert tree.is_descendant("theme-toggle-btn", "prompt-container")
    assert not tree.is_descendant("theme-toggle-btn", "prompt-form")


def test_file_picker_matches_backend_supported_extensions() -> None:
    tree = parse_template()
    assert set(tree.elements["document-file-input"][1]["accept"].split(",")) == SUPPORTED_DOCUMENT_EXTENSIONS


def test_document_polling_only_runs_for_nonterminal_states() -> None:
    assert run_state_function("shouldPollDocuments", [{"status": "ready"}]) is False
    assert run_state_function("shouldPollDocuments", [{"status": "failed"}]) is False
    assert run_state_function("shouldPollDocuments", [{"status": "processing"}]) is True
    assert run_state_function("shouldPollDocuments", [{"status": "deleting"}]) is True


def test_document_actions_follow_document_status() -> None:
    assert run_state_function("documentActions", {"status": "ready"}) == ["download", "delete"]
    assert run_state_function("documentActions", {"status": "failed"}) == ["download", "retry", "delete"]
    assert run_state_function("documentActions", {"status": "processing"}) == ["download", "delete"]
    assert run_state_function("documentActions", {"status": "deleting"}) == []


def test_stream_reducer_routes_deltas_to_its_own_session_buffer() -> None:
    state = {"one": {"text": "A", "status": ""}, "two": {"text": "B", "status": ""}}
    result = run_state_function("reduceStreamEvent", {
        "buffers": state, "sessionId": "two", "event": {"type": "delta", "text": "!"},
    })
    assert result == {"one": {"text": "A", "status": ""}, "two": {"text": "B!", "status": ""}}


def test_short_delta_error_stream_keeps_a_terminal_buffer() -> None:
    program = (
        f"import {{ reduceStreamEvent }} from {json.dumps(STATE.as_uri())};"
        "let buffers = {one: {text: '', status: '', user: 'question'}};"
        "buffers = reduceStreamEvent({buffers, sessionId: 'one', event: {type: 'delta', text: 'answer'}});"
        "buffers = reduceStreamEvent({buffers, sessionId: 'one', event: {type: 'error'}});"
        "console.log(JSON.stringify(buffers.one));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {
        "text": "answer", "status": "Lỗi", "user": "question", "terminal": "error",
    }


def test_script_uses_session_routes_and_independent_upload() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    for text in ("selectedSessionId", "streamControllers", "documentPollTimer", "/api/sessions", "/messages", "/chat", "/stop"):
        assert text in script
    assert "new FormData(documentUploadForm)" in script
    assert "new FormData(promptForm)" not in script
    assert "setInterval(loadDocuments, 1500)" in script
    assert "function renderStream(sessionId) { if (sessionId === selectedSessionId) renderMessages(); }" in script
    assert 'message(role === "assistant" ? "bot" : role, content)' in script
    assert "function syncResponseState()" in script
    assert "document.body.classList.toggle(\"bot-responding\", streamControllers.has(selectedSessionId))" in script


def test_bot_markdown_renderer_formats_content_without_trusting_html() -> None:
    source = "# Tiêu đề\n\n- một\n- hai\n\n**đậm** và `code`\n\n<script>alert(1)</script>"
    program = (
        f"import {{ renderMarkdown }} from {json.dumps(MARKDOWN.as_uri())};"
        f"console.log(JSON.stringify(renderMarkdown({json.dumps(source)})));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        check=True, capture_output=True, text=True,
    )
    rendered = json.loads(result.stdout)
    assert "<h1>Tiêu đề</h1>" in rendered
    assert "<ul><li>một</li><li>hai</li></ul>" in rendered
    assert "<strong>đậm</strong>" in rendered
    assert "<code>code</code>" in rendered
    assert "<script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered


def test_ui_uses_inline_item_menus_without_native_dialogs() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "more_horiz" in script
    assert "session-item-menu" in script
    assert "document-item-menu" in script
    assert "confirm(" not in script
    assert "prompt(" not in script
    assert "alert(" not in script


def test_controls_follow_round_and_pill_shape_rules() -> None:
    style = STYLE.read_text(encoding="utf-8")
    assert ".icon-button" in style and "border-radius: 50%" in style
    assert ".pill-button" in style and "border-radius: 999px" in style
    assert ".prompt-form" in style and "border-radius: 999px" in style
    assert "left: var(--sidebar-width)" not in style
    assert "justify-content: space-between" in style
    assert "min-height: 58px" not in style
    assert "min-height: 52px" in style


def test_browser_script_is_valid_javascript() -> None:
    subprocess.run(["node", "--check", str(SCRIPT)], check=True, capture_output=True, text=True)
