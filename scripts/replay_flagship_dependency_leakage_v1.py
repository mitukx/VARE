"""Test installed-repair leakage and recalibrate a baseline-only dependency image."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import qualify_flagship_development_v1 as qualification  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise ValueError("external integrity protocol anchor mismatch")
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"integrity experiment input drift: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    task_protocol = json.loads((ROOT / protocol["qualification_protocol"]).read_text())
    results = []
    for image_name, image in protocol["images"].items():
        inspect = subprocess.run(["docker", "--context", "colima", "image", "inspect", image],
                                 capture_output=True, check=True, timeout=10)
        (args.output / (image_name+"-image.json")).write_bytes(inspect.stdout)
        for task in task_protocol["tasks"]:
            name = task["id"]
            folder = args.output / image_name / name
            folder.mkdir(parents=True)
            if name == "thinking-markers":
                calls = qualification.thinking_calls()
                delegate = b"from mlx_lm.tokenizer_utils import TokenizerWrapper\n"
            else:
                calls = []
                for case in task_protocol["gptq_cases"]:
                    reference = ROOT / protocol["qualification_run"] / name / (case["id"]+"-reference.json")
                    raw = json.loads(json.loads(reference.read_text())["stdout"])
                    calls.append(qualification.grading.FunctionCall(case["id"], "adapter", "run", [raw["request"]], raw["expected"]))
                delegate = b"from mlx_lm.quant.gptq import gptq_quantize\n"
            variants = {"installed-delegation": delegate}
            if image_name == "baseline-only":
                baseline = qualification.source_bytes(args.source_repository, task["baseline_commit"], task["target"])
                fixed = qualification.source_bytes(args.source_repository, task["fixed_commit"], task["target"])
                if hashlib.sha256(baseline).hexdigest() != task["baseline_sha256"] or hashlib.sha256(fixed).hexdigest() != task["fixed_sha256"]:
                    raise ValueError("upstream module identity mismatch")
                variants.update({"baseline": baseline, "known-fix": fixed})
                for mutant in task["mutations"]:
                    before = mutant["old"].encode()
                    if fixed.count(before) != 1:
                        raise ValueError("ambiguous mutant")
                    variants[mutant["id"]] = fixed.replace(before, mutant["new"].encode(), 1)
            runtime = {**task_protocol, "image": image}
            for variant, content in variants.items():
                started = time.monotonic()
                report = qualification.grade_source(content, (ROOT / task["adapter"]).read_bytes(), calls, runtime)
                row = {"image_arm": image_name, "task": name, "variant": variant,
                       "source_sha256": hashlib.sha256(content).hexdigest(),
                       "elapsed_seconds": time.monotonic()-started, **asdict(report)}
                (folder / (variant+".json")).write_text(json.dumps(row, indent=2, sort_keys=True)+"\n")
                results.append(row)
                print(json.dumps({key:row[key] for key in ["image_arm", "task", "variant", "passed"]}), flush=True)
    lookup = {(row["image_arm"], row["task"], row["variant"]): row["passed"] for row in results}
    reproduced = lookup[("original", "gptq-groups", "installed-delegation")]
    closed = not lookup[("baseline-only", "gptq-groups", "installed-delegation")]
    calibrated = all(row["passed"] == (row["variant"] == "known-fix") for row in results if row["image_arm"] == "baseline-only")
    summary = {"protocol_sha256": args.protocol_sha256, "installed_repair_leakage_reproduced": reproduced,
               "specific_delegation_path_closed": closed, "baseline_only_image_calibrated": calibrated,
               "results": results, "model_runs": 0, "optimizer_steps": 0,
               "task_exposure": "Both tasks remain consumed Development; no model rerun or new held-out evidence."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    return 0 if reproduced and closed and calibrated else 2


if __name__ == "__main__":
    raise SystemExit(main())
