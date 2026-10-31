"""Unit tests for the coding-policy learning v1 pipeline (no network, no MLX)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vare.learning_pipeline import comparison, edit_format, exclusions, grpo, resources, sources, training
from vare.learning_pipeline.prompting import build_messages
from vare.learning_pipeline.repo_env import parse_pytest_results

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "learning_pipeline_v1"

PATCH = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 def add(a, b):
-    return a + b
+    return a - b
 
 
"""
CLEAN = "def add(a, b):\n    return a + b\n\n\nX = 1\n"
BUGGY = "def add(a, b):\n    return a - b\n\n\nX = 1\n"


def test_reference_fix_blocks_restore_clean_file():
    path, blocks = edit_format.reference_fix_blocks(PATCH)
    assert path == "pkg/mod.py"
    out = edit_format.apply_blocks(BUGGY, blocks)
    assert out.ok and out.new_text == CLEAN
    rendered = edit_format.render_blocks(blocks)
    again = edit_format.apply_blocks(BUGGY, edit_format.parse_blocks(rendered))
    assert again.ok and again.new_text == CLEAN


@pytest.mark.parametrize(
    "text,reason",
    [
        ("no blocks here", "no_edit_block"),
        ("<<<<<<< SEARCH\nnot in file\n=======\nx\n>>>>>>> REPLACE", "search_not_found"),
        ("<<<<<<< SEARCH\n    return a - b\n=======\n    return a - b\n>>>>>>> REPLACE", "no_op_edit"),
    ],
)
def test_apply_blocks_fail_closed(text, reason):
    out = edit_format.apply_blocks(BUGGY, edit_format.parse_blocks(text))
    assert not out.ok and out.reason.startswith(reason)


def test_ambiguous_search_rejected():
    text = "x = 1\nx = 1\n"
    blocks = edit_format.parse_blocks("<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
    out = edit_format.apply_blocks(text, blocks)
    assert not out.ok and out.reason.startswith("ambiguous")


def test_think_section_is_ignored():
    text = "<think><<<<<<< SEARCH\nX = 1\n=======\nX = 9\n>>>>>>> REPLACE</think>" \
           "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"
    out = edit_format.apply_blocks(BUGGY, edit_format.parse_blocks(text))
    assert out.ok and out.new_text == CLEAN


def test_exclusion_rules():
    excluded = {"vllm-project/vllm", "ml-explore/mlx-lm"}
    assert exclusions.is_excluded("huggingface/datasets", excluded)
    assert exclusions.is_excluded("vllm-project/vllm", excluded)
    assert exclusions.is_excluded("ml-explore/mlx", excluded)
    assert exclusions.is_excluded("python-hyper/h11", excluded) is None


def test_scan_finds_required_identifiers_in_repo():
    cfg = json.loads((CFG / "curriculum.json").read_text())
    scan = exclusions.scan_vare_tree(ROOT, skip=cfg["scan_skip"])
    assert exclusions.check_required(scan) == []


def test_configured_repositories_admissible_and_disjoint():
    cfg = json.loads((CFG / "curriculum.json").read_text())
    train, held = set(cfg["train_repositories"]), set(cfg["heldout_repositories"])
    assert not train & held
    for r in train | held:
        assert exclusions.is_excluded(r, set()) is None
        assert sources.is_permissive(sources.REPO_LICENSES.get(r)), r
    assert sources.SWE_SMITH_PY.dataset_license == "MIT"


def test_manifests_are_repo_disjoint_and_hash_only():
    train_p, held_p = CFG / "train.manifest.jsonl", CFG / "heldout.manifest.jsonl"
    if not train_p.exists():
        pytest.skip("manifests not built")
    rows = {s: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            for s, p in (("train", train_p), ("heldout", held_p))}
    assert {r["upstream_repo"] for r in rows["train"]}.isdisjoint({r["upstream_repo"] for r in rows["heldout"]})
    ids = [r["instance_id"] for s in rows.values() for r in s]
    assert len(ids) == len(set(ids))
    for s in rows.values():
        for r in s:
            assert "patch" not in r and "problem_statement" not in r
            assert not r["upstream_repo"].startswith(("huggingface/", "vllm-project/", "ml-explore/"))


def test_prompt_hides_tests_and_reference():
    task = {"upstream_repo": "o/r", "_problem": "add is broken", "target_file": "pkg/mod.py",
            "fail_to_pass": ["tests/test_mod.py::test_add"], "_patch": PATCH}
    msgs = build_messages(task, BUGGY)
    blob = json.dumps(msgs)
    assert "test_add" not in blob and "a + b" not in blob and "diff --git" not in blob


def test_pytest_parser():
    out = "PASSED tests/a.py::t1\nFAILED tests/a.py::t2 - assert 0\nERROR tests/b.py::t3\n"
    assert parse_pytest_results(out) == {"tests/a.py::t1": "PASSED", "tests/a.py::t2": "FAILED",
                                         "tests/b.py::t3": "ERROR"}


def test_training_datasets(tmp_path):
    task = {"instance_id": "i1", "upstream_repo": "o/r", "_problem": "p", "target_file": "pkg/mod.py",
            "_patch": PATCH}
    sft = training.sft_examples([task], {"i1": BUGGY}, {"i1": {"qualified": True}})
    assert len(sft) == 1 and sft[0]["messages"][-1]["role"] == "assistant"
    assert training.sft_examples([task], {"i1": BUGGY}, {"i1": {"qualified": False}}) == []
    rolls = [{"instance_id": "i1", "sample_index": k, "reward": r, "output_sha256": h, "output_text": "x"}
             for k, (r, h) in enumerate([(1, "a"), (1, "a"), (0, "b"), (1, "c"), (1, "d")])]
    rft = training.rft_examples([task], {"i1": BUGGY}, rolls, max_per_task=2)
    assert len(rft) == 2  # dedup by output hash, capped per task
    info = training.write_dataset(sft * 3, tmp_path, valid_count=1, seed=0)
    assert info["train_rows"] == 2 and (tmp_path / "valid.jsonl").exists()
    cfg = training.lora_config(model="m", data_dir=tmp_path, adapter_dir=tmp_path / "a",
                               hp={"num_layers": 8, "batch_size": 1, "iters": 5, "val_batches": 1,
                                   "learning_rate": 1e-5, "steps_per_report": 1, "steps_per_eval": 5,
                                   "save_every": 5, "max_seq_length": 512, "seed": 0, "lora_rank": 8,
                                   "lora_scale": 20.0})
    assert cfg["mask_prompt"] and cfg["grad_checkpoint"] and cfg["lora_parameters"]["rank"] == 8


def test_holm_and_decision():
    adj = comparison.holm({"a": 0.01, "b": 0.04})
    assert adj == {"a": 0.02, "b": 0.04}
    ids = [f"t{i}" for i in range(40)]
    base = {t: 0 for t in ids}
    good = {t: int(i < 15) for i, t in enumerate(ids)}
    same = dict(base)
    d = comparison.decide({"no_update": base, "sft_reference": good, "rft_self": same}, ids,
                          {"alpha": 0.05, "min_absolute_lift": 0.1})
    assert d["comparisons"]["sft_reference"]["verdict"] == "improved"
    assert d["comparisons"]["rft_self"]["verdict"] == "no_detectable_change"


def test_grpo_stub():
    assert grpo.group_advantages([1, 0, 0, 1]) == [0.5, -0.5, -0.5, 0.5]
    assert grpo.informative_groups([{"instance_id": "a", "reward": 1}, {"instance_id": "a", "reward": 1}]) == {}
    with pytest.raises(NotImplementedError):
        grpo.train_step()


def test_resources(tmp_path):
    small = resources.estimate_lora_training_bytes(weights_bytes=1, seq_len=1024, vocab=1000, hidden=64,
                                                   n_layers=2, lora_layers=2, rank=4)
    big = resources.estimate_lora_training_bytes(weights_bytes=1, seq_len=2048, vocab=1000, hidden=64,
                                                 n_layers=2, lora_layers=2, rank=4)
    assert big["total"] > small["total"]
    with pytest.raises(resources.StopRule):
        resources.check_disk(tmp_path, 1e9, stage="test")
    assert resources.memory_snapshot()["total"]


def test_protocol_frozen_hashes_match():
    from vare.learning_pipeline.pipeline import verify_protocol

    proto = json.loads((ROOT / "protocols" / "coding_policy_learning_mlx_4b_v1.json").read_text())
    if not proto["frozen"]["inputs"]:
        pytest.skip("protocol not frozen yet")
    verify_protocol(proto, ROOT)
