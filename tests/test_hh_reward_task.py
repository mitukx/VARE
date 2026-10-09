from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    numpy = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hh_reward_task import eligible_pair, metric_summary, paired_bootstrap_interval
from audit_cpu_hh_reward_model import audited_bootstrap, audited_metrics


class CharacterOffsetTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=True):
        return {
            "input_ids": [0] + list(range(1, len(text) + 1)),
            "offset_mapping": [(0, 0)] + [(index, index + 1) for index in range(len(text))],
        }


def test_eligible_pair_uses_exact_shared_context_and_response_offsets():
    row = {
        "chosen": "Human: hello\n\nAssistant: preferred",
        "rejected": "Human: hello\n\nAssistant: rejected",
    }
    parsed = eligible_pair(row, CharacterOffsetTokenizer(), 1024)
    assert parsed is not None
    assert parsed["context"] == "Human: hello\n\nAssistant:"
    assert parsed["chosen_tokens"] == len(" preferred")
    assert parsed["rejected_tokens"] == len(" rejected")
    assert len(parsed["context_hash"]) == 64


def test_invalid_pairs_and_overlength_rows_are_ineligible():
    tokenizer = CharacterOffsetTokenizer()
    mismatched = {
        "chosen": "Human: first\n\nAssistant: yes",
        "rejected": "Human: second\n\nAssistant: no",
    }
    assert eligible_pair(mismatched, tokenizer, 1024) is None
    valid_but_too_long = {
        "chosen": "Human: x\n\nAssistant: " + "a" * 32,
        "rejected": "Human: x\n\nAssistant: " + "b" * 32,
    }
    assert eligible_pair(valid_but_too_long, tokenizer, 24) is None


def test_pair_metrics_use_half_credit_for_exact_ties():
    metrics = metric_summary([0.0])
    assert metrics["pairwise_accuracy"] == 0.5
    assert math.isclose(metrics["logistic_nll"], math.log(2), rel_tol=0, abs_tol=1e-12)
    assert metrics["brier_score"] == 0.25
    assert math.isclose(metrics["expected_calibration_error"], 0.5, rel_tol=0, abs_tol=1e-12)


def test_bootstrap_interval_is_seeded_and_ordered():
    if numpy is None:
        import pytest
        pytest.skip("NumPy is an optional local-only dependency for model-study bootstraps")
    values = [0.1, -0.1, 0.2, 0.0]
    first = paired_bootstrap_interval(values, 20261009, 1000)
    second = paired_bootstrap_interval(values, 20261009, 1000)
    assert first == second
    assert first[0] <= first[1]
    assert first == audited_bootstrap(values, 20261009, 1000)


def test_independent_auditor_metrics_match_from_margins():
    margins = [-2.0, -0.1, 0.0, 0.3, 2.0]
    expected = metric_summary(margins)
    observed = audited_metrics(margins)
    assert expected.keys() == observed.keys()
    for name in expected:
        assert math.isclose(expected[name], observed[name], rel_tol=0, abs_tol=1e-12)
