"""Candidate-visible syntax smoke check."""
import ast
from pathlib import Path
source = Path(__file__).with_name("modeling_musicgen.py").read_text(encoding="utf-8")
tree = ast.parse(source)
assert any(isinstance(node, ast.ClassDef) and node.name == "MusicgenSinusoidalPositionalEmbedding" for node in tree.body)
