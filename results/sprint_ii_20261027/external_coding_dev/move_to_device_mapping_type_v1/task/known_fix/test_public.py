import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

class FakeTensor:
    def __init__(self, device="source"):
        self.device = device
    def to(self, device):
        return FakeTensor(str(device))

class ModelOutputLike(OrderedDict):
    pass

def load_move_to_device():
    tree = ast.parse(Path("pytorch_utils.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "move_to_device")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": SimpleNamespace(Tensor=FakeTensor, device=object)}
    exec(compile(module, "pytorch_utils.py", "exec"), namespace)
    return namespace["move_to_device"]

def test_mapping_subclass_is_preserved_and_values_are_recursively_moved():
    move_to_device = load_move_to_device()
    value = ModelOutputLike([
        ("last_hidden_state", FakeTensor()),
        ("nested", ModelOutputLike([("hidden", FakeTensor())])),
        ("items", [FakeTensor(), (FakeTensor(),)]),
    ])
    moved = move_to_device(value, "cpu:7")
    assert type(moved) is ModelOutputLike
    assert type(moved["nested"]) is ModelOutputLike
    assert moved["last_hidden_state"].device == "cpu:7"
    assert moved["nested"]["hidden"].device == "cpu:7"
    assert moved["items"][0].device == "cpu:7"
    assert moved["items"][1][0].device == "cpu:7"
