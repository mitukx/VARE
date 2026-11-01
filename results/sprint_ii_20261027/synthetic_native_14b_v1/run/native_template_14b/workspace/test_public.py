"""Candidate-visible syntax check for the synthetic fixture."""
import ast
from pathlib import Path
tree = ast.parse(Path(__file__).with_name("mathlib.py").read_text(encoding="utf-8"))
assert any(isinstance(node, ast.FunctionDef) and node.name == "add" for node in tree.body)
