"""Architecture boundaries that prevent revival of the retired runtime."""

import ast
from pathlib import Path


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def test_runtime_has_no_superseded_chat_or_json_compatibility_paths() -> None:
    """Removing a retired module or DTO must not leave a runtime dependency behind."""
    source = Path("src")
    assert not (source / "chat.py").exists()
    assert not (source / "llama.py").exists()

    imports = set().union(*(_imports(path) for path in source.glob("*.py")))
    assert not {"src.chat", "src.llama"}.intersection(imports)

    retired = {"LiveCorpus", "RequestState", "RagIndex", "History", "Corpus"}
    for path in source.glob("*.py"):
        if path.name == "migration.py":
            continue
        names = {
            node.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not retired.intersection(names), path
