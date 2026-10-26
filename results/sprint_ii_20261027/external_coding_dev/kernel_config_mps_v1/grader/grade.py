#!/usr/bin/env python3
"""Independent AST grader for supported device validation paths."""
import ast, json, sys
from pathlib import Path
def main():
    try: tree=ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as exc:
        print(json.dumps({"success":False,"reason":"source_parse_failure","error":str(exc)})); return 0
    constants=[n for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign)) and ((isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="KERNELS_SUPPORTED_DEVICE_TYPES" for t in n.targets)) or (isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=="KERNELS_SUPPORTED_DEVICE_TYPES"))]
    values=[]
    if constants:
        value=constants[0].value
        if isinstance(value,(ast.List,ast.Tuple)): values=[x.value for x in value.elts if isinstance(x,ast.Constant) and isinstance(x.value,str)]
    required={"cuda","rocm","xpu","npu","neuron","tpu","mps"}; functions={n.name:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}; errors=[]
    if not required.issubset(set(values)): errors.append("shared_device_constant_missing_supported_values")
    for name in ("add_to_mapping","add_to_mapping_local","sanitize_kernel_mapping"):
        node=functions.get(name)
        if node is None: errors.append("missing_function:"+name); continue
        uses=any(isinstance(n,ast.Compare) and any(isinstance(op,(ast.In,ast.NotIn)) for op in n.ops) and isinstance(n.left,ast.Name) and n.left.id=="device" and any(isinstance(c,ast.Name) and c.id=="KERNELS_SUPPORTED_DEVICE_TYPES" for c in n.comparators) for n in ast.walk(node))
        if not uses: errors.append("validation_not_using_shared_device_constant:"+name)
    success=not errors; print(json.dumps({"success":success,"reason":"contract_pass" if success else "mps_device_validation_incomplete","errors":errors})); return 0
if __name__=="__main__": raise SystemExit(main())
