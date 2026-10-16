import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_actual_synthetic_pipeline_and_resealed_missing_latency(tmp_path):
    output = tmp_path / "actual-runtime"
    run = subprocess.run([sys.executable, "scripts/benchmark_trajectory_runtime.py", "--trajectories", "20", "--mode", "async", "--sigkill", "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    result = json.loads((output / "summary.json").read_text())
    assert result["pass"] and result["trajectories"] == 20
    assert result["recovery"]["sigkill"] and result["recovery"]["stale_worker_rejected"]
    audited = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(output)], capture_output=True, text=True, timeout=30)
    assert audited.returncode == 0, audited.stderr
    assert json.loads(audited.stdout)["pass"]
    broken = tmp_path / "rehashed-missing-latency"
    shutil.copytree(output, broken, copy_function=os.link)
    samples = (broken / "latencies.jsonl").read_text().splitlines()
    (broken / "latencies.jsonl").unlink()
    (broken / "latencies.jsonl").write_text("\n".join(samples[1:]) + "\n")
    manifest = json.loads((broken / "SHA256.json").read_text())
    manifest["latencies.jsonl"] = hashlib.sha256((broken / "latencies.jsonl").read_bytes()).hexdigest()
    (broken / "SHA256.json").unlink()
    (broken / "SHA256.json").write_text(json.dumps(manifest))
    rejected = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(broken)], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and "missing or duplicated latency samples" in rejected.stderr
    assert len(Path(output / "latencies.jsonl").read_text().splitlines()) == 20
