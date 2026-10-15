"""Read-only reconstruction of checkpoint IO comparison artifacts (stdlib)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as ET


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    assets = manifest["artifact_sha256"]
    require(set(assets) == {str(p.relative_to(root)) for p in root.rglob("*")
                           if p.is_file() and p.name != "manifest.json"}, 'checkpoint evidence check failed')
    for name, expected in assets.items():
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'checkpoint evidence check failed')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, name)
    test_records = json.loads((root / "test-results.json").read_text())
    for label, failures in [("baseline", 7), ("candidate", 0)]:
        cases = ET.parse(root / f"{label}-pytest.xml").findall(".//testcase")
        verdicts = {c.attrib["name"]: "error" if c.find("error") is not None else
                    "fail" if c.find("failure") is not None else
                    "skip" if c.find("skipped") is not None else "pass" for c in cases}
        require(verdicts == test_records[label]["cases"] and len(verdicts) == 15, 'checkpoint evidence check failed')
        counts = Counter(verdicts.values())
        require(counts["fail"] == failures and counts["pass"] == 15 - failures, 'checkpoint evidence check failed')
        require(not counts["error"] and not counts["skip"], 'checkpoint evidence check failed')
        if label == "baseline":
            for name, verdict in verdicts.items():
                target = (name.startswith("test_fifo_") or name.startswith("test_registered_size_")
                          or name.startswith("test_growth_") or name.endswith("promotes[fifo]"))
                require(verdict == ("fail" if target else "pass"), name)
    reconstructed = {}
    for case in ["valid", "oversized", "growth"]:
        reconstructed[case] = {}
        for label in ["baseline", "candidate"]:
            rows = [json.loads((root / f"{label}-{case}-{i}.json").read_text()) for i in range(5)]
            for i, row in enumerate(rows):
                require((row["case"], row["label"], row["repetition"]) == (case, label, i), 'checkpoint evidence check failed')
                require(row["backend_sha256"] == test_records[label]["backend_sha256"], 'checkpoint evidence check failed')
                require(row["wall_s"] > 0 and row["python_peak_bytes"] > 0, 'checkpoint evidence check failed')
                if case == "valid":
                    require(row["verified_bytes"] == 64 * 1024**2 and row["reads"] == 64, 'checkpoint evidence check failed')
                    require(row["rejection"] is None, 'checkpoint evidence check failed')
                    require(row["throughput_mib_s"] == 64 / row["wall_s"], 'checkpoint evidence check failed')
                else:
                    require(row["rejection"] and row["verified_bytes"] == 0, 'checkpoint evidence check failed')
                    if label == "candidate":
                        require(row["python_peak_bytes"] < 512 * 1024, 'checkpoint evidence check failed')
                    if case == "growth":
                        require(row["growth_injected"], 'checkpoint evidence check failed')
            reconstructed[case][label] = {
                f"median_{key}": median(row[key] for row in rows)
                for key in ["wall_s", "python_peak_bytes", "process_peak_rss_bytes", "throughput_mib_s"]
            }
    require(reconstructed == json.loads((root / "summary.json").read_text()), 'checkpoint evidence check failed')
    for label in ["baseline", "candidate"]:
        for fault in ["fifo", "oversized"]:
            evidence = json.loads((root / label / f"recovery-{fault}.json").read_text())
            before, after = evidence["failed_round_status"], evidence["reopened_status"]
            require(before["active_policy_id"] == "p0" and after["active_policy_id"] == "p1", 'checkpoint evidence check failed')
            for status in [before, after]:
                require(status["pending_tx_id"] is None and status["round_owner_id"] is None, 'checkpoint evidence check failed')
                require(status["active_checkpoint_sha256"] == evidence["checkpoint_sha256"][status["active_policy_id"]], 'checkpoint evidence check failed')
            result = evidence["round_result"]
            require(result["promoted_id"] == "p1" and result["decision"]["accepted"], 'checkpoint evidence check failed')
            for key, expected in [("incumbent_eval", 0.5), ("candidate_eval", 1.0)]:
                report = result[key]
                scores = report["metadata"]["per_task_scores"]
                require(len(scores) == report["n"] == 6 and sum(scores.values()) / 6 == expected == report["primary"], 'checkpoint evidence check failed')
                require(report["metadata"]["vare_evaluation_set_sha256"] == evidence["heldout_sha256"], 'checkpoint evidence check failed')
                require(report["metadata"]["vare_checkpoint_sha256"] == evidence["checkpoint_sha256"][report["policy_id"]], 'checkpoint evidence check failed')
            workers = evidence["worker_outcomes"]
            require(len(workers) == 4, 'checkpoint evidence check failed')
            require(sum(bool(w["timeout"]) for w in workers) == int(label == "baseline" and fault == "fifo"), 'checkpoint evidence check failed')
            for report_key in ["incumbent_eval", "candidate_eval"]:
                report = result[report_key]
                require(any(json.loads(w["stdout"]) == report for w in workers
                           if w.get("returncode") == 0), 'checkpoint evidence check failed')
    return {"verified_artifacts": len(assets), "regression_cases_per_arm": 15,
            "measurement_processes": 30, "recovery_records": 4,
            "scope": "same-host consistency reconstruction; no external authentication or reproduction"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle.resolve()), indent=2))
