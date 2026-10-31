"""Static checks of the frozen group-informativeness study protocol (no model)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import grpo_group_informativeness_study as study  # noqa: E402

PROTO = json.loads(study.PROTOCOL.read_text())


def test_p_mixed_matches_closed_form():
    # Beta(1, 5): E[p^4] = 24/3024, E[(1-p)^4] = 5/9
    assert study.p_mixed(0, 4, 4) == pytest.approx(1 - 24 / 3024 - 5 / 9)
    assert study.p_mixed(4, 4, 4) == pytest.approx(study.p_mixed(0, 4, 4))
    assert study.p_mixed(2, 4, 4) > study.p_mixed(1, 4, 4) > study.p_mixed(0, 4, 4)
    assert study.p_mixed(0, 0, 1) == pytest.approx(0.0, abs=1e-12)  # one sample is never mixed


def test_conditions_spend_equal_budgets_on_train_tasks_only():
    budget = PROTO["rollout_budget_per_condition"]
    admitted = json.loads(study.ADMITTED.read_text())
    train, heldout = set(admitted["train"]), set(admitted["heldout"])
    for name, c in PROTO["conditions"].items():
        assert c["group_size"] * len(c["tasks"]) == budget, name
        assert len(set(c["tasks"])) == len(c["tasks"]), name
        assert set(c["tasks"]) <= train and not set(c["tasks"]) & heldout, name
    u4, u8 = PROTO["conditions"]["uniform_g4"]["tasks"], PROTO["conditions"]["uniform_g8"]["tasks"]
    assert u8 == u4[: len(u8)]
    assert "heldout" not in json.dumps(PROTO["prior"]["sources"])


def test_prior_selection_reproduces_from_frozen_estimates():
    pr, seed = PROTO["prior"]["per_task"], PROTO["seed"]
    g = PROTO["conditions"]["prior_g4"]["group_size"]
    ranked = sorted(pr, key=lambda i: (-round(study.p_mixed(pr[i]["k"], pr[i]["n"], g), 12), study.tie_key(seed, i)))
    assert PROTO["conditions"]["prior_g4"]["tasks"] == ranked[: len(PROTO["conditions"]["prior_g4"]["tasks"])]
    for e in pr.values():
        assert e["p_mixed_g4"] == pytest.approx(study.p_mixed(e["k"], e["n"], 4), abs=1e-6)


def test_summary_counts_wasted_generations():
    proto = {"label": "x", "conditions": {"c": {"group_size": 2, "tasks": ["a", "b", "d"]}}}
    groups = [{"condition": "c", "rewards": [0.0, 1.0], "mixed": True, "sec_per_rollout": 2.0},
              {"condition": "c", "rewards": [0.0, 0.0], "mixed": False, "sec_per_rollout": 4.0}]
    s = study.summarize(proto, groups, "wall_clock", 10.0)["conditions"]["c"]
    assert s["groups_done"] == 2 and not s["complete"] and s["wasted_generations"] == 2
    assert s["mixed_group_fraction"] == 0.5 and s["informative_generation_fraction"] == 0.5
    assert s["sec_per_rollout"] == 3.0 and s["all_zero_groups"] == 1 and s["all_one_groups"] == 0
