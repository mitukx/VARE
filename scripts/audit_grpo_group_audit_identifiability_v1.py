#!/usr/bin/env python3
"""Independent replay audit for the frozen GRPO group-audit study."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_group_audit_identifiability_v1.lock.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            value.update(chunk)
    return value.hexdigest()


def labels_for(source: random.Random, world: str) -> tuple[int, int]:
    coin = source.randrange(2)
    return (coin, coin) if world == "A" else (coin, 1 ^ coin)


def recompute_item(seed: int, world: str, choose_random_member: bool) -> int:
    source = random.Random(seed)
    total = 0
    for _sample in range(200):
        pair = labels_for(source, world)
        chosen = source.randrange(2) if choose_random_member else 0
        total += pair[chosen]
    return total


def recompute_group(seed: int, world: str) -> int:
    source = random.Random(seed)
    non_ties = 0
    for _sample in range(100):
        pair = labels_for(source, world)
        non_ties += pair[0] ^ pair[1]
    return non_ties


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    lock = json.loads(PROTOCOL.read_text())
    raw = bundle / "trials.jsonl"
    summary = json.loads((bundle / "summary.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    for filename, expected_hash in manifest.items():
        assert digest(bundle / filename) == expected_hash
    expected_protocol = digest(PROTOCOL)
    assert summary["protocol_id"] == lock["protocol_id"]
    assert summary["protocol_sha256"] == expected_protocol
    assert summary["raw_trials_sha256"] == digest(raw)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    assert len(rows) == 4000
    recovered = {arm: {world: 0 for world in ("A", "B")} for arm in ("item_uniform", "item_proxy_positive", "group_atomic")}
    for row in rows:
        seed, world = row["seed"], row["world"]
        expected_random = recompute_item(seed + 10_000, world, True)
        expected_top = recompute_item(seed + 30_000, world, False)
        expected_mixed = recompute_group(seed + 50_000, world)
        assert row["item_uniform_positive_labels"] == expected_random
        assert row["item_proxy_positive_labels"] == expected_top
        assert row["group_atomic_mixed_groups"] == expected_mixed
        assert row["labels_per_arm"] == 200
        assert row["group_atomic_label_calls"] == 200
        recovered["item_uniform"][world] += int(row["item_uniform_prediction"] == world)
        recovered["item_proxy_positive"][world] += int(row["item_proxy_positive_prediction"] == world)
        recovered["group_atomic"][world] += int(row["group_atomic_prediction"] == world)
    rates_by_world = {
        arm: {world: correct / 2000 for world, correct in outcomes.items()}
        for arm, outcomes in recovered.items()
    }
    rates = {arm: sum(values.values()) / 2 for arm, values in rates_by_world.items()}
    assert rates_by_world == summary["per_world_accuracy"]
    assert rates == summary["balanced_accuracy"]
    assert all(summary[key] == 0.0 for key in ("item_uniform_exact_observation_tv", "item_proxy_positive_exact_observation_tv"))
    assert rates["item_uniform"] <= 0.55
    assert rates["item_proxy_positive"] <= 0.55
    assert rates["group_atomic"] == 1.0
    assert summary["group_signal"]["world_A_advantages"] == [0.0, 0.0]
    assert summary["group_signal"]["world_B_advantages"] == [[1.0, -1.0], [-1.0, 1.0]]
    assert summary["decision"] == "mechanism_supported"
    audit_record = {
        "audit": "PASS",
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": expected_protocol,
        "auditor_sha256": digest(Path(__file__).resolve()),
        "raw_trials_sha256": digest(raw),
        "batches": len(rows),
        "balanced_accuracy": rates,
        "decision": summary["decision"],
    }
    audit_path = bundle / "audit.json"
    audit_path.write_text(json.dumps(audit_record, indent=2, sort_keys=True) + "\n")
    manifest["audit.json"] = digest(audit_path)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit_record))


if __name__ == "__main__":
    main()
