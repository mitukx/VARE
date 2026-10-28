"""Candidate-visible syntax smoke test; semantic checks remain hidden."""
import ast
from pathlib import Path

source = Path(__file__).with_name("modeling_musicgen_melody.py").read_text(encoding="utf-8")
tree = ast.parse(source)
assert any(isinstance(node, ast.ClassDef) and node.name == "MusicgenMelodySinusoidalPositionalEmbedding" for node in tree.body)
