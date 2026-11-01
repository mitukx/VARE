import ast
from pathlib import Path
def test_forward_comprehensions_keep_outer_encoder_tensor():
 tree=ast.parse(Path("modeling_superpoint.py").read_text(encoding="utf-8")); clazz=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=="SuperPointForKeypointDetection"); method=next(n for n in ast.walk(clazz) if isinstance(n,ast.FunctionDef) and n.name=="forward"); comps=[n for n in ast.walk(method) if isinstance(n,ast.ListComp)]
 for name in ("keypoint_decoder","descriptor_decoder"):
  comp=next(c for c in comps if isinstance(c.elt,ast.Call) and isinstance(c.elt.func,ast.Attribute) and c.elt.func.attr==name)
  assert any(isinstance(gen.target,ast.Name) and gen.target.id=="image_hidden_state" for gen in comp.generators)
