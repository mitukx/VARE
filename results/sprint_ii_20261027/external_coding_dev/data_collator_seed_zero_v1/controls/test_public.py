import ast
from pathlib import Path
def test_seed_zero_uses_explicit_none_checks_in_both_rng_paths():
 tree=ast.parse(Path("data_collator.py").read_text(encoding="utf-8")); cls=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=="DataCollatorForLanguageModeling"); funcs={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
 for name in ("torch_call","numpy_call"):
  assert any(isinstance(n,ast.Compare) and isinstance(n.left,ast.Attribute) and isinstance(n.left.value,ast.Name) and n.left.value.id=="self" and n.left.attr=="seed" and len(n.ops)==1 and isinstance(n.ops[0],ast.IsNot) and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value is None for n in ast.walk(funcs[name]))
