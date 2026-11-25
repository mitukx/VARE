#!/usr/bin/env python3
"""Read-only v2 native-sampler diagnostic on an explicitly new task-free cohort."""
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

PROTOCOL = ROOT / "protocols/grpo_pretrained_sampling_alignment_v2.json"


def load_protocol():
    p = json.loads(PROTOCOL.read_text())
    if (p["schema"] != "vare.grpo.pretrained_sampling_alignment/v2"
            or p["sampling_score_mode"] != G.NATIVE_SAMPLER_SCORE_MODE
            or p["tolerance"] != G.ON_POLICY_LOG_RATIO_TOLERANCE
            or p["prior_protocol_sha256"] != D.digest(D.PROTOCOL)):
        raise ValueError("invalid v2 diagnostic protocol or changed v1 protocol")
    return p


def run(p, snapshot: Path, arm: str, output: Path):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("diagnostic must be offline")
    mx.set_default_device(mx.cpu)
    start = time.monotonic()
    files = D.verify_snapshot(snapshot, p)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
    ).strip():
        raise ValueError("source tree must be clean")
    record = {
        "schema": "vare.grpo.pretrained_sampling_alignment_result/v2",
        "source_commit": source, "protocol_sha256": D.digest(PROTOCOL),
        "arm": arm, "snapshot_files": files, "device": str(mx.default_device()),
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "mlx", "mlx-lm", "numpy", "transformers", "huggingface-hub", "safetensors"
        )},
        "optimizer_updated": False, "task_success_measured": False,
        "heldout_accessed": False, "status": "INCOMPLETE", "groups": [],
    }
    try:
        policy, trainable, dtypes, quantized = D.prepare_policy(p, snapshot, arm)
        model = policy.model
        before = D.parameter_digest(model, mx, tree_flatten)
        record.update(parameter_digest_before=before, trainable=trainable,
                      dtype_counts=dtypes, quantized_modules=quantized)
        cfg = G.GRPOConfig(
            temperature=p["temperature"], group_size=len(p["seeds"]),
            sampling_score_mode=p["sampling_score_mode"],
        )
        token_count = 0
        for context in p["contexts"]:
            if time.monotonic() - start > p["limits"]["maximum_wall_seconds_per_arm"]:
                raise TimeoutError("diagnostic wall budget exceeded")
            samples, raw = D.collect_group(policy, p, context, before)
            samples_before = copy.deepcopy(samples)
            alignment = G.audit_on_policy_sampling_alignment(
                model, samples, cfg, policy_checkpoint_id=before, policy_step=0,
                max_action_segments=p["limits"]["max_action_segments_per_group"],
                max_completion_tokens=p["limits"]["max_completion_tokens_per_group"],
                max_sequence_tokens=p["max_sequence_tokens"], tolerance=p["tolerance"],
            )
            control = D.arithmetic_control(model, samples, p, mx)
            if samples != samples_before:
                raise RuntimeError("diagnostic altered captured samples")
            token_count += alignment["n_completion_tokens"]
            if token_count > p["limits"]["maximum_generated_tokens_per_arm"]:
                raise ValueError("diagnostic token budget exceeded")
            legacy = []
            for action, score in zip(raw, control, strict=True):
                legacy.extend(abs(a - b) for a, b in zip(
                    action["sampled_logprobs"], score["uncached_trainer_logprobs"], strict=True,
                ))
            row = {
                "context": context["id"], "alignment": alignment, "actions": raw,
                "legacy_arithmetic_control": control,
                "legacy_sampled_vs_plain": G._alignment_metric(legacy, tolerance=p["tolerance"]),
            }
            record["groups"].append(row)
            print("VARE_NATIVE_ALIGNMENT_GROUP " + json.dumps({
                "arm": arm, "context": context["id"],
                "tokens": alignment["n_completion_tokens"],
                "metrics": alignment["metrics"],
                "legacy_sampled_vs_plain": row["legacy_sampled_vs_plain"],
                "would_reject_if_selected": alignment["would_reject_lag_zero_update"],
            }, sort_keys=True, allow_nan=False), flush=True)
        after = D.parameter_digest(model, mx, tree_flatten)
        record["parameter_digest_after"] = after
        if before != after or model.training:
            raise RuntimeError("model parameters or evaluation mode changed")
        record["parameter_bytes_unchanged"] = True
        record["generated_tokens"] = token_count
        record["status"] = "PASS" if all(
            not group["alignment"]["would_reject_lag_zero_update"] for group in record["groups"]
        ) else "NO_GO_NUMERICAL_MISMATCH"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - start
        record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with output.open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print("VARE_NATIVE_ALIGNMENT_FINAL " + json.dumps({
            "arm": arm, "status": record["status"], "source_commit": source,
            "protocol_sha256": record["protocol_sha256"], "result_sha256": D.digest(output),
            "generated_tokens": record.get("generated_tokens"),
            "parameter_bytes_unchanged": record.get("parameter_bytes_unchanged", False),
            "wall_seconds": record["wall_seconds"], "error": record.get("error"),
        }, sort_keys=True), flush=True)
    if record["status"] != "PASS":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--snapshot-path-file", type=Path, required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--snapshot", type=Path, required=True)
    measure.add_argument("--arm", choices=("fp32", "bf16"), required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = load_protocol()
    if args.command == "download":
        D.download(p, args.snapshot_path_file)
    else:
        run(p, args.snapshot, args.arm, args.output)


if __name__ == "__main__":
    main()
