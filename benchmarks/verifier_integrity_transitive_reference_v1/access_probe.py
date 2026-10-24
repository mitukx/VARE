"""Read-only source and bytecode provenance probe; never grades candidates."""

from __future__ import annotations

import dis
import hashlib
import importlib.metadata
import json
import marshal
from pathlib import Path
from types import CodeType


ROOT = Path("/usr/local/lib/python3.12/site-packages/transformers/models/mimo_v2_flash")
SOURCE_NAMES = ("modeling_mimo_v2_flash.py", "modular_mimo_v2_flash.py")
BYTECODE_NAMES = (
    "modeling_mimo_v2_flash.cpython-312.pyc",
    "modular_mimo_v2_flash.cpython-312.pyc",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(code: CodeType):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from walk(value)


def main() -> None:
    sources = []
    for name in SOURCE_NAMES:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        hits = [
            {"line": index + 1, "text": line.strip()}
            for index, line in enumerate(text.splitlines())
            if "v_scale" in line or "attention_value_scale" in line
        ]
        sources.append({"name": name, "bytes": path.stat().st_size, "sha256": digest(path), "hits": hits})

    bytecode = []
    for name in BYTECODE_NAMES:
        path = ROOT / "__pycache__" / name
        data = path.read_bytes()
        module = marshal.loads(data[16:])
        code_objects = [code for code in walk(module) if "v_scale" in code.co_names]
        instructions = [
            {"function": code.co_name, "op": item.opname, "arg": str(item.argval)}
            for code in code_objects
            for item in dis.get_instructions(code)
            if item.argval == "v_scale" or item.opname == "BINARY_OP"
        ]
        bytecode.append({
            "name": name,
            "bytes": len(data),
            "sha256": digest(path),
            "has_v_scale_code": bool(code_objects),
            "instructions": instructions,
        })

    print(json.dumps({
        "distribution": "transformers",
        "version": importlib.metadata.version("transformers"),
        "source_directory": str(ROOT),
        "sources": sources,
        "bytecode": bytecode,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
