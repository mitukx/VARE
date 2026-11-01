import ast
from pathlib import Path
def test_mps_is_declared_and_used_in_all_validation_paths():
    tree=ast.parse(Path("kernel_config.py").read_text(encoding="utf-8"))
    const=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="KERNELS_SUPPORTED_DEVICE_TYPES" for t in n.targets))
    assert "mps" in [x.value for x in const.elts if isinstance(x,ast.Constant)]
    funcs={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    for name in ("add_to_mapping","add_to_mapping_local","sanitize_kernel_mapping"):
        assert any(isinstance(n,ast.Compare) and any(isinstance(op,(ast.In,ast.NotIn)) for op in n.ops) and isinstance(n.left,ast.Name) and n.left.id=="device" and any(isinstance(x,ast.Name) and x.id=="KERNELS_SUPPORTED_DEVICE_TYPES" for x in n.comparators) for n in ast.walk(funcs[name]))
