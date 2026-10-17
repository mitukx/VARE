"""Independent raw-artifact checks for the synthetic trajectory campaign."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vare.evidence import sha256_file, sha256_json
from vare.trajectory_store import TrajectoryStore, audit_trajectory_store


def require(value, reason):
    if not value:
        raise ValueError(reason)


def audit(root):
    manifest = json.loads((root / "SHA256.json").read_text())
    for name, expected in manifest.items():
        path = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "unsafe artifact path")
        require(path.is_file() and sha256_file(path) == expected, f"artifact mismatch: {name}")
    summary = json.loads((root / "summary.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    require(provenance["protocol_sha256"] == sha256_json(protocol), "protocol mismatch")
    condition = provenance["condition"]
    require(protocol["schema"] == "vare.trajectory-runtime-systems.v2"
            and condition["seed"] in protocol["seeds"] and condition["workers"] == protocol["workers"]
            and condition["trajectories"] in {protocol["stage_a_trajectories"], protocol["stage_b_trajectories"], protocol["optional_stage_c_trajectories"]}
            and condition["mode"] in protocol["modes"] and condition["sigkill"] is True,
            "condition differs from fixed protocol")
    require(all(summary[key] == condition[key] for key in ("seed", "workers", "mode", "trajectories"))
            and summary["verifiers"] == protocol["verifiers"] and summary["recovery"]["sigkill"] is True,
            "summary/condition mismatch")
    for name, expected in provenance["source_files"].items():
        require(sha256_file(root / "source" / name) == expected, "source snapshot mismatch")
    result = audit_trajectory_store(root / "snapshot.db")
    require(result == summary["audit"], "store audit differs from retained summary")
    store = TrajectoryStore(root / "snapshot.db")
    count, steps, latencies, verification = 0, 0, [], []
    # Independent per-step truth checks; never call the producer/build_step.
    with (root / "latencies.jsonl").open() as records, store.connection() as db:
        for line in records:
            sample = json.loads(line)
            row = store.trajectory(sample["trajectory_id"])
            require(row["state"] == "consumed" and row["head"] == sample["trace_sha256"], "latency row is not a consumed trajectory")
            cursor, total = 0, 0
            for step in store.iter_steps(row["id"]):
                require(step["index"] == cursor and step["observation"] == {"value": cursor + 1}, "synthetic observation differs from ground truth")
                total += cursor + 1
                require(step["state"] == {"sum": total} and step["generation"]["response"] == str(total), "incorrect resumable counter state")
                require(step["generation"]["metadata"]["response_token_ids"] == [total] and step["generation"]["metadata"]["response_token_logprobs"] == [0.], "synthetic behavior record mismatch")
                require(step["generation"]["prompt"] == f"Return the final counter {total}" and step["generation"]["metadata"]["prompt_token_ids"] == [0, total], "synthetic conditioning/state mismatch")
                cursor += 1
            require(cursor == row["expected_steps"] == sample["steps"], "incorrect step coverage")
            value = json.loads(row["verification"])
            require(value["score"] == 1. and value["passed"] is True, "incorrect independent synthetic verification")
            require(value["metadata"]["verification_wall_s"] == sample["verification_s"], "verification latency mismatch")
            latencies.append(sample["e2e_s"])
            verification.append(sample["verification_s"])
            count, steps = count + 1, steps + cursor
        recorded_ids = db.execute("SELECT count(*) FROM trajectories WHERE state='consumed'").fetchone()[0]
        require(count == recorded_ids == summary["trajectories"], "missing or duplicated latency samples")
        # UNIQUE complete coverage is checked separately, even after rehashing.
        seen = set()
        with (root / "latencies.jsonl").open() as records:
            for line in records:
                ident = json.loads(line)["trajectory_id"]
                require(ident not in seen, "duplicate latency trajectory")
                seen.add(ident)
        require(steps == summary["steps"], "step numerator mismatch")
        for key, values in [("e2e_latency_s", latencies), ("verification_latency_s", verification)]:
            values.sort()
            for percentile in (50, 95, 99):
                expected = values[math.ceil(len(values) * percentile / 100) - 1]
                require(summary[key][f"p{percentile}"] == expected, "latency quantile mismatch")
        require(summary["timed_events"] == result["events"] - 2, "event numerator mismatch")
        for name, numerator in [("trajectories_per_s", count), ("steps_per_s", steps), ("events_per_s", summary["timed_events"])]:
            require(summary[name] == numerator / summary["wall_s"], "throughput mismatch")
        if summary["recovery"]["sigkill"]:
            require(summary["recovery"]["exitcode"] == -9 and summary["recovery"]["stale_worker_rejected"] is True, "missing SIGKILL/fencing outcome")
            attempts = [tuple(row) for row in db.execute("SELECT generation,state FROM attempts WHERE trajectory_id='g000000-0' AND stage='execute' ORDER BY generation")]
            require(attempts == [(1, "lease_expired"), (2, "done")], "worker recovery lineage mismatch")
    return {"pass": True, "trajectories": count, "steps": steps, "files_verified": len(manifest),
            "events": result["events"], "effective_duplicates": result["effective_duplicates"],
            "scope": "Rehashes saved bytes, streams step truth/state and verifies latency/throughput derivations. MAC admission was checked online; no signing key is published. No model optimizer or promotion audit."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(audit(parser.parse_args().output), indent=2))
