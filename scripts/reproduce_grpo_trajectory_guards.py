"""Replay two synthetic behavior-probability defects against a pinned baseline.

No pretrained weights, protected tasks, model generation, or remote fetches.
Run from a checkout containing BASELINE_COMMIT; stdout is the evidence JSON.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

BASELINE_COMMIT = "287621875c08c22419810e021f9180f0d830b259"
BASELINE_SHA256 = "09410c9cafe7a4b62af8eca0462254ece6177d4bc42f72323c148a4ddc5c188f"
ROOT = Path(__file__).resolve().parents[1]


def _prepared(module):
    import mlx.core as mx
    import mlx.nn as nn

    class Bigram(nn.Module):
        def __init__(self):
            super().__init__()
            self.table = nn.Embedding(64, 64)

        def __call__(self, tokens):
            return self.table(tokens)

    mx.random.seed(0)
    model = Bigram()
    mx.eval(model.parameters())
    samples = []
    for index, reward in enumerate((1.0, 0.0)):
        action = [index + 1]
        samples.append(module.Sample(
            instance_id="synthetic-two-turn-task", behavior_checkpoint_id="policy-0",
            behavior_step=0, prompt_ids=[10], completion_ids=action,
            reward=reward, verifier="synthetic-verifier", sample_index=index,
            trajectory_segments=[
                module.TrajectorySegment(prompt_ids=[10], completion_ids=list(action)),
                module.TrajectorySegment(prompt_ids=[10, action[0], 30 + index],
                                         completion_ids=list(action)),
            ],
        ))
    cfg = module.GRPOConfig(group_size=2)
    module.compute_behavior_logprobs(model, samples, cfg)
    samples[1].trajectory_segments[1].behavior_logprobs[0] -= 0.5
    for sample in samples:
        sample.meta.update(behavior_logprob_source="sampling_distribution", behavior_group_size=2,
                           behavior_temperature=1.0, behavior_top_p=1.0)
    return model, samples, cfg


def _observe(module):
    import mlx.core as mx
    import mlx.optimizers as optim

    model, samples, cfg = _prepared(module)
    before_scores = copy.deepcopy([sample.trajectory_segments for sample in samples])
    module.compute_behavior_logprobs(model, samples, cfg)
    overwritten = before_scores != [sample.trajectory_segments for sample in samples]
    # Independently check train_step without rescoring the corrupted denominator.
    model, samples, cfg = _prepared(module)
    optimizer = optim.Adam(learning_rate=0.1)
    before = module.trainable_digest(model)
    rejected, reason = False, None
    try:
        module.train_step(model, optimizer, samples, cfg,
                          policy_checkpoint_id="policy-0", policy_step=0)
    except module.StalePolicyError as exc:
        rejected, reason = True, str(exc)
    mx.eval(model.parameters(), optimizer.state)
    return {
        "recorded_scores_overwritten": overwritten,
        "mismatched_lag_zero_rejected": rejected,
        "parameters_changed": module.trainable_digest(model) != before,
        "optimizer_step": int(optimizer.state["step"].item()),
        "rejection": reason,
    }


def main():
    import mlx.core as mx
    from vare import grpo_mlx as current

    source = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:src/vare/grpo_mlx.py"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        raise ValueError("baseline source hash mismatch")
    mx.set_default_device(mx.cpu)
    with tempfile.TemporaryDirectory(prefix="grpo-guard-baseline-") as directory:
        path = Path(directory) / "baseline.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("grpo_guard_baseline", path)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        observations = {"baseline": _observe(baseline), "patched": _observe(current)}
    old, new = observations["baseline"], observations["patched"]
    expected = (
        old["recorded_scores_overwritten"] and not old["mismatched_lag_zero_rejected"]
        and old["parameters_changed"] and old["optimizer_step"] == 1
        and not new["recorded_scores_overwritten"] and new["mismatched_lag_zero_rejected"]
        and not new["parameters_changed"] and new["optimizer_step"] == 0
    )
    result = {
        "schema": "vare_grpo_trajectory_guard_replay_v1",
        "scope": "synthetic forced-action defect replay; no model capability claim",
        "baseline_commit": BASELINE_COMMIT, "baseline_source_sha256": BASELINE_SHA256,
        "patched_source_sha256": hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        "reproducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "mlx": mx.__version__,
                    "device": str(mx.default_device()), "platform": platform.system(),
                    "machine": platform.machine()},
        "fixture": {"seed": 0, "group_size": 2, "action_segments": 4,
                    "later_action_logprob_perturbation": -0.5},
        "observations": observations, "expected_regression_reproduced": expected,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not expected:
        raise SystemExit("expected baseline defect or patched rejection was not reproduced")


if __name__ == "__main__":
    main()
