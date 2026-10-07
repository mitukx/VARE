from __future__ import annotations

from pathlib import Path
import subprocess

from vare.environments import EngineeringTaskSpec, ExecutableEvaluator


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "benchmarks/historical/rvl_behavior_policy_parity/task.json"


def _workspace(root: Path, *, fixed: bool) -> Path:
    target = root / ("fixed" if fixed else "broken")
    (target / "src/rvl_systems").mkdir(parents=True)
    backend = '"sampling_temperature": temperature\nGenerationConfig(\n"top_k": 0\n"top_p": 1.0\n' if fixed else '"top_k": 0\n"top_p": 1.0\n'
    trainer = 'sampling_temperature\nresponse_logits.float() / temperature\ngreedy scores are not behavior probabilities\n' if fixed else 'response_logits.float()\n'
    (target / "src/rvl_systems/hf_backend.py").write_text(backend, encoding="utf-8")
    (target / "src/rvl_systems/hf_trainer.py").write_text(trainer, encoding="utf-8")
    return target


def test_historical_external_evaluator_distinguishes_contract(tmp_path):
    spec = EngineeringTaskSpec.load(SPEC_PATH)
    evaluator = ExecutableEvaluator(spec, evaluator_root=SPEC_PATH.parent)
    broken = evaluator.evaluate(_workspace(tmp_path, fixed=False))
    fixed = evaluator.evaluate(_workspace(tmp_path, fixed=True))
    assert not broken.passed
    assert fixed.passed
