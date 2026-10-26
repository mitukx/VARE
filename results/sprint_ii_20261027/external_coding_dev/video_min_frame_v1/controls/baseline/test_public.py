"""Public smoke check for syntax and the expected entry point."""
import ast
from pathlib import Path

source = Path(__file__).with_name("video_utils.py").read_text(encoding="utf-8")
tree = ast.parse(source)
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "default_sample_indices_fn"]
assert len(functions) == 1, "default_sample_indices_fn must remain defined once"
