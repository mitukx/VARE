"""Read-only follow-up audit for adjacent Transformers MiMo implementations."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import marshal
from pathlib import Path
from types import CodeType


ROOT = Path("/usr/local/lib/python3.12/site-packages/transformers/models/mimo_v2_flash")
TOKENS = ("attention_value_scale", "v_scale")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_names(code: CodeType) -> set[str]:
    names = set(code.co_names)
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(code_names(value))
    return names


def main() -> None:
    files = []
    for path in sorted(ROOT.glob("*.py")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        hits = []
        for i, line in enumerate(lines):
            if any(token in line for token in TOKENS):
                hits.append({
                    "line": i + 1,
                    "text": line.strip(),
                    "context": [
                        {"line": j + 1, "text": lines[j].rstrip()}
                        for j in range(max(0, i - 2), min(len(lines), i + 4))
                    ],
                })
        files.append({"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size, "hits": hits})
    bytecode_files = []
    for path in sorted((ROOT / "__pycache__").glob("*.pyc")):
        data = path.read_bytes()
        try:
            names = code_names(marshal.loads(data[16:]))
            parse_error = None
        except Exception as exc:
            names = set()
            parse_error = type(exc).__name__
        bytecode_files.append({
            "path": str(path),
            "sha256": digest(path),
            "size_bytes": len(data),
            "contains_attention_value_scale": "attention_value_scale" in names,
            "contains_v_scale": "v_scale" in names,
            "parse_error": parse_error,
        })
    print(json.dumps({
        "distribution": "transformers",
        "version": metadata.version("transformers"),
        "model_directory": str(ROOT),
        "torch_import_spec_available": importlib.util.find_spec("torch") is not None,
        "files": files,
        "bytecode_files": bytecode_files,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
