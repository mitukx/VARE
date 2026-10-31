"""Invariants of the frozen Train-pool probe protocol (no model, no repos)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import grpo_train_pool_probe as probe  # noqa: E402

PROTO = json.loads(probe.PROTOCOL.read_text())


def _manifest(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_classification_rule():
    assert [probe.classify(k, 4) for k in range(5)] == ["all_zero", "mixed", "mixed", "mixed", "all_one"]


def test_candidates_are_train_only_and_unused():
    cur = json.loads(probe.CURRICULUM.read_text())
    heldout = _manifest(probe.HELDOUT_MANIFEST)
    train = _manifest(probe.TRAIN_MANIFEST)
    prior = json.loads(probe.PRIOR_PROTOCOL.read_text())["prior"]["per_task"]
    ids = [c["instance_id"] for c in PROTO["candidates"]]
    assert len(ids) == len(set(ids)) > 0
    assert all(c["split"] == "train" and c["upstream_repo"] in cur["train_repositories"] for c in PROTO["candidates"])
    assert not {c["upstream_repo"] for c in PROTO["candidates"]} & set(cur["heldout_repositories"])
    assert not set(ids) & {t["instance_id"] for t in heldout}
    assert not set(ids) & set(prior)
    hpatch = {t["patch_sha256"] for t in heldout + train}
    assert not {c["patch_sha256"] for c in PROTO["candidates"]} & hpatch


def test_protocol_pins_inputs():
    assert PROTO["base_protocol"]["sha256"] == probe.sha256_file(probe.BASE_PROTOCOL)
    assert PROTO["existing_pool"]["source"]["sha256"] == probe.sha256_file(probe.PRIOR_PROTOCOL)
    assert PROTO["stop_rules"]["swap_used_fraction_max"] == 0.97
    assert PROTO["stop_rules"]["free_disk_gib_min"] == 8.0
    assert PROTO["decoding"]["group_size"] == 4 and PROTO["decoding"]["temperature"] == 0.7
