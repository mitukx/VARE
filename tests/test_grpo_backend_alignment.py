"""Backend preflight contracts without MLX or a model download."""
from types import SimpleNamespace

import pytest

from scripts import diagnose_grpo_backend_alignment as D


class FakeMX:
    gpu = "gpu"
    cpu = "cpu"

    def __init__(self, available):
        self.metal = SimpleNamespace(is_available=lambda: available)
        self.selected = []
        self.evaluated = []

    def set_default_device(self, device):
        self.selected.append(device)

    def default_device(self):
        return self.selected[-1]

    def array(self, values):
        class Array:
            def __mul__(self, scale):
                return SimpleNamespace(tolist=lambda: [x * scale for x in values])

        return Array()

    def eval(self, value):
        self.evaluated.append(value)


def test_metal_unavailable_never_falls_back_to_cpu():
    spec = D.load_protocol()["arms"]["metal-bf16"]
    mx = FakeMX(False)
    assert D.select_backend(spec, mx, system="Darwin", machine="arm64") == {
        "status": "BLOCKED_BACKEND", "reason": "metal_unavailable",
    }
    assert mx.selected == mx.evaluated == []


@pytest.mark.parametrize("arm,system,machine,device", [
    ("metal-q4", "Darwin", "arm64", "gpu"),
    ("cpu-q4", "Linux", "x86_64", "cpu"),
])
def test_registered_backend_selects_and_evaluates_requested_device(arm, system, machine, device):
    spec = D.load_protocol()["arms"][arm]
    mx = FakeMX(True)
    assert D.select_backend(spec, mx, system=system, machine=machine) == {
        "status": "READY", "device": device,
    }
    assert mx.selected == [device] and len(mx.evaluated) == 1


def test_wrong_architecture_is_blocked_before_device_selection():
    spec = D.load_protocol()["arms"]["metal-q4"]
    mx = FakeMX(True)
    result = D.select_backend(spec, mx, system="Darwin", machine="x86_64")
    assert result["status"] == "BLOCKED_BACKEND"
    assert mx.selected == []
