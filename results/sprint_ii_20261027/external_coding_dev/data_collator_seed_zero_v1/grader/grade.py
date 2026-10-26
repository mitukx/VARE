#!/usr/bin/env python3
"""Independent AST grader for seeded torch/numpy collator paths."""
import ast,json,sys
from pathlib import Path
def main():
 try: tree=ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
 except (OSError,UnicodeError,SyntaxError) as e: print(json.dumps({"success":False,"reason":"source_parse_failure","error":str(e)})); return 0
 cls=next((n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=="DataCollatorForLanguageModeling"),None); funcs={n.name:n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))} if cls else {}; errors=[]
 for name in ("torch_call","numpy_call"):
  node=funcs.get(name); ok=False
  if node:
   for n in ast.walk(node):
    if isinstance(n,ast.Compare) and isinstance(n.left,ast.Attribute) and isinstance(n.left.value,ast.Name) and n.left.value.id=="self" and n.left.attr=="seed" and len(n.ops)==1 and isinstance(n.ops[0],ast.IsNot) and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value is None: ok=True
  if not ok: errors.append("seed_none_check_missing:"+name)
 success=not errors; print(json.dumps({"success":success,"reason":"contract_pass" if success else "seed_zero_not_supported","errors":errors})); return 0
if __name__=="__main__": raise SystemExit(main())
