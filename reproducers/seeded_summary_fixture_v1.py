"""Hand-made task/output grid; never generate confirmation-cohort tasks."""
from vare import format_vs_correctness_v2 as legacy


def synthetic_grid(patterns=None):
    patterns = patterns or {"format_sft": [4, 0, 0, 0, 0],
                            "correctness_reward_rl": [1, 1, 1, 0, 0]}
    confirm = [{"task_id": "synthetic-a", "expression": "1 + 1", "split": "confirm"},
               {"task_id": "synthetic-b", "expression": "2 + 1", "split": "confirm"}]
    pack = [{"task_id": "synthetic-pack", "expression": "3 + 1", "split": "pack"}]
    answers = {"synthetic-a": 2, "synthetic-b": 3, "synthetic-pack": 4}
    rows = [{**unit, "loss": 1.0} for unit in legacy.iter_train_units()]
    for unit in legacy.iter_eval_units(confirm, pack):
        seed_index = legacy.RUN_SEEDS.index(unit["run_seed"])
        successes = patterns.get(unit["arm"], [0] * 5)[seed_index]
        correct = unit["sample_index"] < successes
        value = answers[unit["task_id"]] + (0 if correct else 100)
        rows.append({**unit, "output": f"#### {value}"})
    return confirm, pack, rows
