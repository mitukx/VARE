import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest


@pytest.mark.parametrize("options", [
    ["--seed", "73011", "--sigkill"],
    ["--workers", "1", "--sigkill"],
    ["--trajectories", "20", "--sigkill"],
    [],
])
def test_undeclared_protocol_condition_fails_before_execution(tmp_path, options):
    output = tmp_path / "invalid"
    run = subprocess.run([sys.executable, "scripts/benchmark_trajectory_runtime.py", "--trajectories", "100", "--output", str(output), *options], capture_output=True, text=True, timeout=10)
    assert run.returncode != 0
    assert not (output / "live.db").exists()
    assert not (output / "summary.json").exists()


def test_actual_synthetic_pipeline_and_resealed_missing_latency(tmp_path):
    output = tmp_path / "actual-runtime"
    run = subprocess.run([sys.executable, "scripts/benchmark_trajectory_runtime.py", "--trajectories", "100", "--mode", "async", "--sigkill", "--output", str(output)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    result = json.loads((output / "summary.json").read_text())
    assert result["pass"] and result["trajectories"] == 100
    assert result["recovery"]["sigkill"] and result["recovery"]["stale_worker_rejected"]
    audited = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(output), "--unanchored-new-run"], capture_output=True, text=True, timeout=30)
    assert audited.returncode == 0, audited.stderr
    assert json.loads(audited.stdout)["pass"]
    invalid_condition = tmp_path / "rehashed-condition"
    shutil.copytree(output, invalid_condition, copy_function=os.link)
    provenance_path = invalid_condition / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["condition"]["seed"] = 73011
    provenance_path.unlink()
    provenance_path.write_text(json.dumps(provenance))
    hashes = json.loads((invalid_condition / "SHA256.json").read_text())
    hashes["provenance.json"] = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    (invalid_condition / "SHA256.json").unlink()
    (invalid_condition / "SHA256.json").write_text(json.dumps(hashes))
    rejected_condition = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(invalid_condition), "--unanchored-new-run"], capture_output=True, text=True, timeout=30)
    assert rejected_condition.returncode != 0 and "condition differs from fixed protocol" in rejected_condition.stderr
    for change in ("allowed_seed", "mode", "protocol"):
        relabeled = tmp_path / ("relabeled-" + change)
        shutil.copytree(output, relabeled, copy_function=os.link)
        altered = {name: json.loads((relabeled / name).read_text()) for name in ("summary.json", "provenance.json", "protocol.json")}
        if change == "mode":
            altered["summary.json"]["mode"] = altered["provenance.json"]["condition"]["mode"] = "sync"
        else:
            seed = 75012 if change == "allowed_seed" else 73011
            altered["summary.json"]["seed"] = altered["provenance.json"]["condition"]["seed"] = seed
            if change == "protocol":
                altered["protocol.json"]["seeds"].append(seed)
                from vare.evidence import sha256_json
                altered["provenance.json"]["protocol_sha256"] = sha256_json(altered["protocol.json"])
        hashes = json.loads((relabeled / "SHA256.json").read_text())
        for name, data in altered.items():
            (relabeled / name).unlink()
            (relabeled / name).write_text(json.dumps(data))
            hashes[name] = hashlib.sha256((relabeled / name).read_bytes()).hexdigest()
        (relabeled / "SHA256.json").unlink()
        (relabeled / "SHA256.json").write_text(json.dumps(hashes))
        rejected_label = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(relabeled), "--unanchored-new-run"], capture_output=True, text=True, timeout=30)
        assert rejected_label.returncode != 0 and ("frozen protocol" in rejected_label.stderr or "snapshot condition" in rejected_label.stderr)
    external = tmp_path / "trusted-original-manifest.json"
    external.write_bytes((output / "SHA256.json").read_bytes())
    anchored = subprocess.run([sys.executable,"scripts/audit_trajectory_runtime.py",str(output),"--expected-manifest",str(external)],capture_output=True,text=True,timeout=30)
    assert anchored.returncode==0 and json.loads(anchored.stdout)["headline_timing_anchored"]
    wall_tamper = tmp_path / "inflated-rate"
    shutil.copytree(output,wall_tamper,copy_function=os.link)
    data = json.loads((wall_tamper/"summary.json").read_text())
    data["wall_s"] /= 10
    for key in ("trajectories_per_s","steps_per_s","events_per_s"):
        data[key] *= 10
    (wall_tamper/"summary.json").unlink()
    (wall_tamper/"summary.json").write_text(json.dumps(data))
    hashes = json.loads((wall_tamper/"SHA256.json").read_text())
    hashes["summary.json"] = hashlib.sha256((wall_tamper/"summary.json").read_bytes()).hexdigest()
    (wall_tamper/"SHA256.json").unlink()
    (wall_tamper/"SHA256.json").write_text(json.dumps(hashes))
    denied = subprocess.run([sys.executable,"scripts/audit_trajectory_runtime.py",str(wall_tamper),"--expected-manifest",str(external)],capture_output=True,text=True,timeout=30)
    assert denied.returncode!=0 and "external artifact anchor mismatch" in denied.stderr
    broken = tmp_path / "rehashed-missing-latency"
    shutil.copytree(output, broken, copy_function=os.link)
    samples = (broken / "latencies.jsonl").read_text().splitlines()
    (broken / "latencies.jsonl").unlink()
    (broken / "latencies.jsonl").write_text("\n".join(samples[1:]) + "\n")
    manifest = json.loads((broken / "SHA256.json").read_text())
    manifest["latencies.jsonl"] = hashlib.sha256((broken / "latencies.jsonl").read_bytes()).hexdigest()
    (broken / "SHA256.json").unlink()
    (broken / "SHA256.json").write_text(json.dumps(manifest))
    rejected = subprocess.run([sys.executable, "scripts/audit_trajectory_runtime.py", str(broken), "--unanchored-new-run"], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and "missing or duplicated latency samples" in rejected.stderr
    assert len(Path(output / "latencies.jsonl").read_text().splitlines()) == 100
