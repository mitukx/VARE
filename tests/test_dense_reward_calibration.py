import json
from pathlib import Path

from scripts.calibrate_verifier_reward import rewards


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "benchmarks/verifier_consistent_async_grpo_v7/reward-calibration-v1.json").read_text())
SPEC = PROTOCOL["reward_definition"]


def test_dense_verifier_transition_changes_meaningful_surface_rewards():
    exact = rewards("leaf on the", "earth", "ground", SPEC)
    related = rewards("leaf on the", "earth", "grass", SPEC)
    unrelated = rewards("leaf on the", "earth", "water", SPEC)
    unknown = rewards("leaf on the", "earth", "page", SPEC)

    assert exact == {"v0": 1.0, "v1": 1.0, "surface_class": "earth"}
    assert related == {"v0": 0.9, "v1": 0.6, "surface_class": "vegetation"}
    assert unrelated == {"v0": 0.65, "v1": 0.25, "surface_class": "water"}
    assert unknown == {"v0": 0.0, "v1": 0.0, "surface_class": "unknown"}


def test_surface_extractor_uses_first_whole_word_surface():
    result = rewards("prompt", "earth", "grass beside the ground", SPEC)
    assert result["surface_class"] == "vegetation"
    assert result["v0"] == 0.9
    assert result["v1"] == 0.6
