from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vare.heldout_exact_integer import evaluate_expression, load_task_pack
from vare.integer_seed_split import (
    NGRAM_N,
    build_split,
    canonical_split_bytes,
    derive_seeds,
    ngram_overlap,
    task_from_seed,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "benchmarks/heldout/integer_seed_split_v1/splits.json"
PROTOCOL = ROOT / "benchmarks/heldout/integer_seed_split_v1/protocol.json"
V1_PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"


def test_committed_seeds_match_the_master_and_the_protocol_hash():
    assert SPLIT_FILE.read_bytes() == canonical_split_bytes()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["splits_sha256"] == hashlib.sha256(SPLIT_FILE.read_bytes()).hexdigest()
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == (
        "690b855b27416ed16fbed9456fb812cef6f9b6f7c31d1d34425a940b3191f322"
    )
    document = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    for name in ("train", "dev", "confirm"):
        assert document[f"{name}_seeds"] == list(derive_seeds(name))
        assert derive_seeds(name) == derive_seeds(name)


def test_splits_are_deterministic_disjoint_and_absent_from_v1():
    splits = {name: build_split(name) for name in ("train", "dev", "confirm")}
    again = build_split("confirm")
    assert [task.expression for task in splits["confirm"]] == [task.expression for task in again]
    for left, right in (("train", "confirm"), ("train", "dev"), ("dev", "confirm")):
        assert ngram_overlap(
            [task.expression for task in splits[left]],
            [task.expression for task in splits[right]],
            NGRAM_N,
        ) == set()
    v1 = {row["expression"] for row in load_task_pack(V1_PACK.read_text(encoding="utf-8"))}
    ours = {task.expression for tasks in splits.values() for task in tasks}
    assert ours.isdisjoint(v1)
    for task in splits["confirm"]:
        assert evaluate_expression(task.expression) == task.value
        assert task.expression in task.prompt


def test_difficulty_parameter_changes_the_expression_shape():
    easy = task_from_seed(derive_seeds("train")[0], "train", difficulty=1)
    hard = task_from_seed(derive_seeds("train")[0], "train", difficulty=3)
    assert easy.expression.count("+") == 1 and "-" not in easy.expression
    assert "*" in hard.expression
    assert evaluate_expression(hard.expression) == hard.value
