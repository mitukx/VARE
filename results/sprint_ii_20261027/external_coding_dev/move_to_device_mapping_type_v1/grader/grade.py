#!/usr/bin/env python3
"""Independent behavioral grader for recursive mapping-type preservation."""
import ast
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

class FakeTensor:
    def __init__(self, device="source"):
        self.device = device
    def to(self, device):
        return FakeTensor(str(device))

class OutputMap(OrderedDict):
    pass

def main(path):
    try:
        tree=ast.parse(Path(path).read_text(encoding="utf-8"))
        node=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=="move_to_device")
        scope={"torch":SimpleNamespace(Tensor=FakeTensor,device=object)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),path,"exec"),scope)
        move=scope["move_to_device"]
        value=OutputMap([("outer",OutputMap([("tensor",FakeTensor())])),("sequence",(FakeTensor(),))])
        result=move(value,"cpu:9")
        ordinary=move({"tensor":FakeTensor()},"cpu:9")
        ok=(type(result) is OutputMap and type(result["outer"]) is OutputMap
            and result["outer"]["tensor"].device=="cpu:9"
            and result["sequence"][0].device=="cpu:9"
            and type(ordinary) is dict and ordinary["tensor"].device=="cpu:9")
        print(json.dumps({"success":ok,"reason":"recursive_mapping_type_contract_pass" if ok else "mapping_type_or_recursive_move_failed"}))
    except Exception as exc:
        print(json.dumps({"success":False,"reason":"grader_execution_failure","error":f"{type(exc).__name__}: {exc}"}))
    return 0
if __name__=="__main__": raise SystemExit(main(sys.argv[1]))
