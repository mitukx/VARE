"""Public syntax smoke test; this does not test the hidden task semantics."""
import ast
from pathlib import Path

source = Path("/workspace/src/transformers/tokenization_utils_base.py")
ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
print("PASS: candidate source parses")
