#!/usr/bin/env python3
"""Independent AST grader for SuperPoint compile-safe comprehension bindings."""
import ast,json,sys
from pathlib import Path
def main():
 try: tree=ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
 except (OSError,UnicodeError,SyntaxError) as e: print(json.dumps({"success":False,"reason":"source_parse_failure","error":str(e)})); return 0
 clazz=next((n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=="SuperPointForKeypointDetection"),None); method=next((n for n in ast.walk(clazz) if isinstance(n,ast.FunctionDef) and n.name=="forward"),None) if clazz else None; errors=[]
 if method is None: errors.append("missing_superpoint_forward")
 else:
  comps=[n for n in ast.walk(method) if isinstance(n,ast.ListComp)]
  def target(c,name): return any(any(isinstance(n,ast.Name) and n.id==name for n in ast.walk(g.target)) for g in c.generators)
  keypoint=next((c for c in comps if isinstance(c.elt,ast.Call) and isinstance(c.elt.func,ast.Attribute) and c.elt.func.attr=="keypoint_decoder"),None)
  descriptor=next((c for c in comps if isinstance(c.elt,ast.Call) and isinstance(c.elt.func,ast.Attribute) and c.elt.func.attr=="descriptor_decoder"),None)
  if keypoint is None or not target(keypoint,"image_hidden_state"): errors.append("keypoint_comprehension_binding_invalid")
  if descriptor is None or not target(descriptor,"image_hidden_state"): errors.append("descriptor_comprehension_binding_invalid")
  if descriptor and not any(isinstance(n,ast.Name) and n.id=="last_hidden_state" for gen in descriptor.generators for n in ast.walk(gen.iter)): errors.append("descriptor_iteration_no_longer_uses_outer_hidden_states")
 success=not errors; print(json.dumps({"success":success,"reason":"contract_pass" if success else "compiled_forward_binding_incomplete","errors":errors})); return 0
if __name__=="__main__": raise SystemExit(main())
