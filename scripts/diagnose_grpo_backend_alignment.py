#!/usr/bin/env python3
"""Prospective read-only native-sampler backend diagnostic."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import diagnose_grpo_pretrained_sampling as D  # noqa: E402
from vare import grpo_mlx as G  # noqa: E402

PROTOCOL = ROOT / "protocols/grpo_pretrained_sampling_alignment_v3.json"
PRIOR = ROOT / "protocols/grpo_pretrained_sampling_alignment_v2.json"


def load_protocol():
    p = json.loads(PROTOCOL.read_text())
    if (p["schema"] != "vare.grpo.pretrained_sampling_alignment/v3"
            or p["sampling_score_mode"] != G.NATIVE_SAMPLER_SCORE_MODE
            or p["tolerance"] != G.ON_POLICY_LOG_RATIO_TOLERANCE
            or p["prior_protocol_sha256"] != D.digest(PRIOR)):
        raise ValueError("invalid backend protocol or changed v2 protocol")
    return p


def select_backend(spec, mx, *, system, machine):
    """Never substitute CPU for Metal or silently run on an unregistered host."""
    if system != spec["system"] or machine not in spec["machines"]:
        return {"status": "BLOCKED_BACKEND", "reason": "platform_not_registered"}
    if spec["device"] == "gpu":
        if not mx.metal.is_available():
            return {"status": "BLOCKED_BACKEND", "reason": "metal_unavailable"}
        mx.set_default_device(mx.gpu)
    elif spec["device"] == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        raise ValueError("unknown requested device")
    # Evaluate actual work on the selected default backend before model access.
    probe = mx.array([1.0, 2.0]) * 2
    mx.eval(probe)
    if probe.tolist() != [2.0, 4.0]:
        raise RuntimeError("backend arithmetic preflight failed")
    return {"status": "READY", "device": str(mx.default_device())}


def preflight(p, arm):
    import mlx.core as mx

    spec = p["arms"][arm]
    result = select_backend(spec, mx, system=platform.system(), machine=platform.machine())
    return {
        **result, "arm": arm, "requested_device": spec["device"],
        "precision": spec["precision"], "platform": platform.platform(),
        "machine": platform.machine(), "mlx": importlib.metadata.version("mlx"),
        "protocol_sha256": D.digest(PROTOCOL),
    }


def write_json(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(p, snapshot: Path, arm: str, output: Path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("diagnostic must be offline")
    host = preflight(p, arm)
    if host["status"] != "READY":
        raise RuntimeError("requested backend is blocked; no CPU fallback")
    started = time.monotonic()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
    ).strip():
        raise ValueError("source tree must be clean")
    record = {
        "schema": "vare.grpo.pretrained_sampling_alignment_result/v3",
        "source_commit": source, "protocol_sha256": D.digest(PROTOCOL),
        "arm": arm, "backend": host, "snapshot_files": D.verify_snapshot(snapshot, p),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "mlx", "mlx-lm", "numpy", "transformers", "huggingface-hub", "safetensors"
        )},
        "optimizer_updated": False, "task_success_measured": False,
        "heldout_accessed": False, "status": "INCOMPLETE", "groups": [],
    }
    try:
        print("VARE_BACKEND_STAGE preparing " + arm, flush=True)
        policy, trainable, dtypes, quantized = D.prepare_policy(
            p, snapshot, p["arms"][arm]["precision"],
        )
        model = policy.model
        before = D.parameter_digest(model, mx, tree_flatten)
        record.update(parameter_digest_before=before, trainable=trainable,
                      dtype_counts=dtypes, quantized_modules=quantized,
                      preparation_seconds=time.monotonic() - started)
        print("VARE_BACKEND_STAGE collecting " + arm, flush=True)
        cfg = G.GRPOConfig(temperature=p["temperature"], group_size=len(p["seeds"]),
                           sampling_score_mode=p["sampling_score_mode"])
        context, = p["contexts"]
        samples, raw = D.collect_group(policy, p, context, before)
        samples_before = copy.deepcopy(samples)
        if time.monotonic() - started > p["limits"]["maximum_wall_seconds_per_arm"]:
            raise TimeoutError("diagnostic wall budget exceeded")
        print("VARE_BACKEND_STAGE differentiating " + arm, flush=True)
        alignment = G.audit_on_policy_sampling_alignment(
            model, samples, cfg, policy_checkpoint_id=before, policy_step=0,
            max_action_segments=p["limits"]["max_action_segments_per_group"],
            max_completion_tokens=p["limits"]["max_completion_tokens_per_group"],
            max_sequence_tokens=p["max_sequence_tokens"], tolerance=p["tolerance"],
        )
        if samples != samples_before:
            raise RuntimeError("diagnostic altered captured samples")
        if (alignment["n_action_segments"] != 4
                or alignment["n_completion_tokens"] != p["limits"]["maximum_generated_tokens_per_arm"]):
            raise ValueError("incomplete or oversized diagnostic group")
        record["groups"].append({"context": context["id"], "actions": raw, "alignment": alignment})
        after = D.parameter_digest(model, mx, tree_flatten)
        record["parameter_digest_after"] = after
        if before != after or model.training:
            raise RuntimeError("parameter bytes or evaluation mode changed")
        if time.monotonic() - started > p["limits"]["maximum_wall_seconds_per_arm"]:
            raise TimeoutError("diagnostic wall budget exceeded")
        record["parameter_bytes_unchanged"] = True
        record["generated_tokens"] = alignment["n_completion_tokens"]
        record["status"] = ("NO_GO_NUMERICAL_MISMATCH" if alignment["would_reject_lag_zero_update"]
                            else "PASS")
        print("VARE_BACKEND_ALIGNMENT_GROUP " + json.dumps({
            "arm": arm, "tokens": record["generated_tokens"],
            "metrics": alignment["metrics"], "would_reject": alignment["would_reject_lag_zero_update"],
            "quantized_modules": quantized,
        }, sort_keys=True, allow_nan=False), flush=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["process_peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record["rss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
        write_json(output, record)
        print("VARE_BACKEND_ALIGNMENT_FINAL " + json.dumps({
            "arm": arm, "status": record["status"], "source_commit": source,
            "protocol_sha256": record["protocol_sha256"], "result_sha256": D.digest(output),
            "generated_tokens": record.get("generated_tokens"),
            "parameter_bytes_unchanged": record.get("parameter_bytes_unchanged", False),
            "wall_seconds": record["wall_seconds"], "error": record.get("error"),
        }, sort_keys=True), flush=True)


def main():
    p = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("preflight")
    probe.add_argument("--arm", choices=p["arms"], required=True)
    probe.add_argument("--output", type=Path, required=True)
    download = sub.add_parser("download")
    download.add_argument("--snapshot-path-file", type=Path, required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--snapshot", type=Path, required=True)
    measure.add_argument("--arm", choices=p["arms"], required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(p, args.arm)
        write_json(args.output, result)
        print("VARE_BACKEND_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    elif args.command == "download":
        D.download(p, args.snapshot_path_file)
    else:
        run(p, args.snapshot, args.arm, args.output)


if __name__ == "__main__":
    main()
