"""Record nested evaluator pytest availability with and without VARE .venv on PATH."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

from vare.environments import EngineeringTaskSpec, ExecutableEvaluator, Workspace

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "benchmarks/smoke/stable_logsumexp/task.json"
ORACLE = ROOT / "tests/fixtures/oracle/fix_stable_logsumexp.py"
VENV_BIN = str(ROOT / ".venv/bin")


@contextmanager
def path_for_case(include_venv: bool):
    previous = os.environ.get("PATH", "")
    parts = [part for part in previous.split(os.pathsep) if part != VENV_BIN]
    if include_venv:
        parts.insert(0, VENV_BIN)
    os.environ["PATH"] = os.pathsep.join(parts)
    try:
        yield
    finally:
        os.environ["PATH"] = previous


def run_case(include_venv: bool):
    spec = EngineeringTaskSpec.load(SPEC)
    with Workspace(spec) as workspace:
        subprocess.run([sys.executable, str(ORACLE), str(workspace.path)], check=True)
        with path_for_case(include_venv):
            result = ExecutableEvaluator(spec).evaluate(workspace.path)
    return result.to_dict()


for name, include in (("without_venv_path", False), ("with_venv_path", True)):
    output = Path(__file__).with_name(name + ".json")
    output.write_text(json.dumps(run_case(include), indent=2) + "\n")
    print(name, "passed=", json.loads(output.read_text())["passed"])
