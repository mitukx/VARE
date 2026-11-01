"""Public smoke checks: parse the edited configuration without importing it."""
import ast
from pathlib import Path

source = Path(__file__).with_name("configuration_csm.py").read_text(encoding="utf-8")
tree = ast.parse(source)
classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CsmConfig"]
assert len(classes) == 1, "CsmConfig must remain defined once"
methods = [node for node in classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__post_init__"]
assert len(methods) == 1, "CsmConfig.__post_init__ must remain defined once"
