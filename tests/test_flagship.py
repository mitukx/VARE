from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from minigrpo.expr import evaluate_expression
from minigrpo.flagship import (
    PRIOR_SEC_EVAL,
    PRIOR_SEC_TRAIN,
    assign_random_rewards,
    assign_rewards,
    batch_seed,
    fit_plan,
    fixture_heldout,
    normalize_completions,
    parse_gsm_gold,
    pick_difficulty,
    render_heldout_svg,
    render_rsi_svg,
    run_cpu_smoke,
    scripted_self_verifier_hack,
    wilson_interval,
)
from minigrpo.grpo import incorrect_reward
from minigrpo.splits import build_split, derive_seeds, task_from_seed

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_flagship.py"
NOTEBOOK = ROOT / "notebooks" / "flagship_a100.ipynb"


def test_wilson_zero_of_sixty_four_matches_the_known_bound() -> None:
    low, high = wilson_interval(0, 64)
    assert low == 0.0
    assert math.isclose(high, 0.05662405979280534, abs_tol=1e-15)
    assert wilson_interval(64, 64)[1] == 1.0


def test_difficulty_four_keeps_the_difficulty_two_row() -> None:
    task = task_from_seed(derive_seeds("train")[0], "train", difficulty=2)
    assert task.expression == "23544 + 19091 - 10702"
    assert task.value == 31933
    harder = build_split("train", difficulty=4)
    assert len(harder) == 128
    assert evaluate_expression(harder[0].expression) == harder[0].value
    confirm = set(derive_seeds("confirm"))
    used = {row.seed for row in harder} | {row.seed for row in build_split("dev", difficulty=4)}
    assert used.isdisjoint(confirm)


def test_picker_prefers_the_band_then_the_nearest_rate() -> None:
    chosen, inside = pick_difficulty(((2, 0.8), (3, 0.4), (4, 0.1)))
    assert chosen == 3 and inside
    chosen, inside = pick_difficulty(((2, 0.9), (3, 0.8), (4, 0.7)))
    assert chosen == 4 and not inside


def test_random_rewards_match_the_rate_and_ignore_the_text() -> None:
    draws = []
    for step in range(400):
        rewards = assign_random_rewards("qwen25-1.5b", 20261031, step, 8, 0.3)
        assert rewards == assign_random_rewards("qwen25-1.5b", 20261031, step, 8, 0.3)
        assert len(rewards) == 8
        draws.extend(rewards)
    rate = sum(draws) / len(draws)
    assert abs(rate - 0.3) < 0.03
    texts = ["#### 1", "#### 1"]
    assert assign_rewards("incorrect", "1 + 1", texts) == [1.0, 1.0]
    assert assign_rewards("correctness", "1 + 1", ["#### 2"]) == [1.0]
    assert incorrect_reward("1 + 1", "nope") == 1.0
    assert normalize_completions(["#### 1", " "]) == ["#### 1", "(empty)"]
    try:
        normalize_completions(["", "  "])
    except RuntimeError:
        pass
    else:
        raise AssertionError("an empty batch was kept")


def test_budget_fit_reduces_full_steps_with_the_unmeasured_prior() -> None:
    plan = fit_plan("full", PRIOR_SEC_TRAIN, PRIOR_SEC_EVAL, "prior")
    assert plan["steps"] == 60
    assert plan["heldout_n"] == 200
    assert float(plan["eta_seconds"]) <= 9000.0
    assert int(plan["steps"]) % int(plan["eval_every"]) == 0
    assert "unmeasured priors" in " ".join(plan["notes"])
    fast = fit_plan("full", 0.05, 0.05, "measured")
    assert fast["steps"] == 100
    slow = fit_plan("full", 2.0, 2.0, "measured")
    assert slow["steps"] == 40
    assert slow["heldout_n"] == 40
    assert "eta_exceeds_budget" in slow["notes"]
    assert "reduced held-out" in " ".join(slow["notes"])


def test_cpu_smoke_resumes_without_duplicating_rows(tmp_path: Path) -> None:
    first = run_cpu_smoke(tmp_path)
    metrics = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in metrics.splitlines()]
    train = [row for row in rows if row["kind"] == "train"]
    assert len(train) == 5 * 2
    for row in train:
        for key in (
            "reward_mean",
            "entropy_mean",
            "response_length_mean",
            "kl_mean",
            "format_compliance_rate",
            "apply_update",
            "group_size",
        ):
            assert key in row
        assert row["group_size"] == 2
    assert {row["arm"] for row in train} == {"correctness", "random", "incorrect", "format", "base"}
    assert all(row["apply_update"] is False for row in train if row["arm"] == "base")
    evals = [row for row in rows if row["kind"] == "eval"]
    assert evals
    assert all("wilson_low" in row and "wilson_high" in row for row in evals)
    confirm = set(derive_seeds("confirm"))
    assert all(int(row["task_seed"]) not in confirm for row in train)
    hack = json.loads((tmp_path / "scripted_hack.json").read_text(encoding="utf-8"))
    assert hack["accepted"] is False
    assert "verifier_disagreement" in hack["reasons"]
    assert "insufficient_primary_gain" in hack["reasons"]
    assert "heldout_divergence" in hack["signals"]
    assert hack["source"] == "cpu_fixture"
    rsi = [row for row in rows if row["kind"] == "rsi_round"]
    assert len(rsi) == 2
    assert {row["reward_source"] for row in rsi} == {"ground_truth", "self_verifier"}
    svg = render_heldout_svg(rows)
    assert "held-out strict accuracy" in svg
    assert "reject" in render_rsi_svg(rows) or "accept" in render_rsi_svg(rows)
    assert fixture_heldout(2)[0].source == "fixture"
    assert parse_gsm_gold("The answer is #### 1,000") == 1000
    assert batch_seed(7, 2, 1) == batch_seed(7, 2, 1)
    second = run_cpu_smoke(tmp_path)
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == metrics
    assert len(second["heldout"]) == len(first["heldout"])
    direct = scripted_self_verifier_hack()
    assert direct["accepted"] is False


def test_plan_and_cpu_smoke_do_not_import_torch(tmp_path: Path) -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    flagship = (ROOT / "minigrpo" / "flagship.py").read_text(encoding="utf-8")
    assert "import torch" not in script
    assert "import datasets" not in script
    assert "import torch" not in flagship.split("def ", 1)[0]
    assert "import datasets" not in flagship.split("def ", 1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    planned = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "full", "--plan"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert planned.returncode == 0, planned.stderr
    payload = json.loads(planned.stdout)
    assert payload["steps"] == 60
    assert payload["rate_source"] == "prior"
    smoked = subprocess.run(
        [sys.executable, str(SCRIPT), "--cpu-smoke", "--output", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert smoked.returncode == 0, smoked.stderr
    assert "heldout_rows" in smoked.stdout


def test_notebook_points_at_the_flagship_runner() -> None:
    document = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in document["cells"])
    assert 'drive.mount("/content/drive")' in source
    assert "Qwen/Qwen2.5-1.5B-Instruct" in source
    assert "meta-llama/Llama-3.2-1B-Instruct" in source
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in source
    assert "989aa7980e4cf806f80c7fef2b1adb7bc71aa306" in source
    assert "scripts/run_flagship.py" in source
    assert "cursor/minigrpo-colab-af75" in source
    assert "2.5" in source
    assert 'MODE = "full"' in source
