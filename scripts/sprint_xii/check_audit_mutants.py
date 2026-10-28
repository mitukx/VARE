"""Reproduce three auditor binding regressions on disposable evidence copies."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(protocol, source, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    script = Path(__file__).parent
    results = []
    cases = [
        ("old-pointer", "active-final-commit-path"),
        ("wrong-gradient-origin", "gradient-origin-state"),
        ("evaluation-mean", "evaluation-aggregation"),
    ]
    for name, expected in cases:
        run = output / name
        shutil.copytree(source, run)
        arm = run / "seed-101" / "A"
        if name == "old-pointer":
            m = json.loads((arm / "checkpoint-0.pt.gz.json").read_text())
            (arm / "active.json").write_text(
                json.dumps(
                    {
                        "path": "checkpoint-0.pt.gz",
                        "sha256": m["sha256"],
                        "policy_version": m["metadata"]["policy_version"],
                    }
                )
            )
        elif name == "wrong-gradient-origin":
            path = arm / "learner.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            event = next(
                x for x in rows if x["kind"] == "optimizer_applied" and x["group"] == 1
            )
            event["bias_before4"] = [x + 0.125 for x in event["bias_before4"]]
            previous = "0" * 64
            for row in rows:
                row.pop("sha")
                row["previous"] = previous
                row["sha"] = hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                previous = row["sha"]
            path.write_text(
                "".join(
                    json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n"
                    for x in rows
                )
            )
        else:
            path = arm / "evaluation.json"
            d = json.loads(path.read_text())
            d["mean_v1_reward"] += 0.25
            path.write_text(json.dumps(d))
        row = {"mutation": name, "expected_v2_failure": expected}
        old = subprocess.run(
            [
                sys.executable,
                str(script / "audit_async_verifier.py"),
                "--protocol",
                str(protocol),
                "--output",
                str(run),
            ],
            capture_output=True,
            text=True,
        )
        (output / (name + "-v1.log")).write_text(old.stdout + old.stderr)
        row["v1_returncode"] = old.returncode
        receipt = output / (name + "-v2.json")
        new = subprocess.run(
            [
                sys.executable,
                str(script / "audit_async_verifier_v2.py"),
                "--protocol",
                str(protocol),
                "--output",
                str(run),
                "--receipt",
                str(receipt),
            ],
            capture_output=True,
            text=True,
        )
        (output / (name + "-v2.log")).write_text(new.stdout + new.stderr)
        failure = json.loads(receipt.with_suffix(".failure.json").read_text())
        row.update(v2_returncode=new.returncode, failure=failure)
        assert new.returncode != 0 and expected in failure["message"]
        results.append(row)
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.protocol, a.source, a.output)
