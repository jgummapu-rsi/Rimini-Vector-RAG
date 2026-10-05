"""Export the import closure of our Docling entry points from a pinned Git clone.

Usage: python -m scripts.vendor_docling /path/to/docling-clone
The export excludes upstream tests, docs, CLI, examples and Git history. It keeps
imports inside functions/optional branches because upstream plugin registration
can reach them even when their inference engines are disabled.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REVISION = "caba660f6ff3a2ee8df39bd39d478046749f815c"
VERSION = "2.133.0"
ROOT = Path(__file__).resolve().parents[1]


def export(clone: Path) -> None:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(clone), *args])

    if git("rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError(f"Checkout upstream revision {REVISION} first")
    files = git("ls-tree", "-r", "--name-only", REVISION, "docling").decode().splitlines()
    modules = {
        name.removesuffix(".py").replace("/", ".").removesuffix(".__init__"): name
        for name in files
        if name.endswith(".py")
    }
    pending = [
        "docling.document_converter",
        "docling.models.plugins.defaults",
        "docling.utils.model_downloader",
    ]
    selected: dict[str, bytes] = {}
    while pending:
        module = pending.pop()
        name = modules.get(module)
        if not name or name in selected:
            continue
        content = git("show", f"{REVISION}:{name}")
        selected[name] = content
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        package = module if name.endswith("/__init__.py") else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(content)):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                    base = ".".join([*parent, *([base] if base else [])])
                pending.append(base)
                pending.extend(f"{base}.{alias.name}" for alias in node.names)
    # Runtime resources are not discoverable through Python imports.
    for name in files:
        if not name.endswith(".py") and "/.agents/" not in name:
            selected[name] = git("show", f"{REVISION}:{name}")
    selected["LICENSE"] = git("show", f"{REVISION}:LICENSE")
    selected["UPSTREAM_README.md"] = git("show", f"{REVISION}:README.md")
    destination = ROOT / "vendor" / "docling"
    for name, content in sorted(selected.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {
        "repository": "https://github.com/docling-project/docling.git",
        "revision": REVISION,
        "version": VERSION,
        "selection": "static import closure including optional imports, plus runtime resources",
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in sorted(selected.items())
        },
    }
    (destination / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(selected)} upstream files to {destination}")


if __name__ == "__main__":
    export(Path(sys.argv[1]))
