#!/usr/bin/env python3
"""Bounded real-pretrained-model sampling alignment; never runs a task or optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.long_horizon_eval import TraceableMLXPolicy, _sampling_token_logprob  # noqa: E402
from vare import grpo_mlx as G  # noqa: E402

PROTOCOL = ROOT / "protocols/grpo_pretrained_sampling_alignment_v1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_protocol():
    p = json.loads(PROTOCOL.read_text())
    if p["schema"] != "vare.grpo.pretrained_sampling_alignment/v1":
        raise ValueError("unexpected diagnostic protocol")
    if p["tolerance"] != G.ON_POLICY_LOG_RATIO_TOLERANCE:
        raise ValueError("protocol tolerance differs from the production guard")
    return p


def download(p, destination: Path):
    from huggingface_hub import snapshot_download

    if destination.exists():
        raise FileExistsError(destination)
    model = p["model"]
    snapshot = Path(snapshot_download(
        repo_id=model["repo_id"], revision=model["revision"], token=False,
        allow_patterns=[
            model["weight_file"], "config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
        ],
    ))
    if snapshot.name != model["revision"]:
        raise ValueError("download did not resolve to the pinned commit")
    verify_snapshot(snapshot, p)
    destination.write_text(str(snapshot.resolve()) + "\n")


def verify_snapshot(snapshot: Path, p):
    weight = snapshot / p["model"]["weight_file"]
    if not weight.is_file() or weight.stat().st_size > p["limits"]["maximum_model_weight_bytes"]:
        raise ValueError("missing or oversized pinned weights")
    if digest(weight) != p["model"]["weight_sha256"]:
        raise ValueError("pretrained model weight SHA-256 mismatch")
    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "llama" or config.get("model_file") or config.get("auto_map"):
        raise ValueError("unexpected model architecture or remote code")
    files = {
        path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(snapshot.iterdir()) if path.is_file()
    }
    return files


def parameter_digest(model, mx, tree_flatten):
    """Hash exact tensor bytes, shapes and dtypes, independent of header ordering."""
    with tempfile.TemporaryDirectory(prefix="vare-align-weights-") as tmp:
        path = Path(tmp) / "parameters.safetensors"
        mx.save_safetensors(str(path), dict(sorted(tree_flatten(model.parameters()))))
        h = hashlib.sha256()
        with path.open("rb") as stream:
            header_len = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_len))
            start = 8 + header_len
            for name, info in sorted(header.items()):
                if name == "__metadata__":
                    continue
                h.update(json_digest([name, info["dtype"], info["shape"]]).encode())
                left, right = info["data_offsets"]
                stream.seek(start + left)
                remaining = right - left
                while remaining:
                    data = stream.read(min(1024 * 1024, remaining))
                    if not data:
                        raise EOFError("truncated parameter digest snapshot")
                    h.update(data)
                    remaining -= len(data)
        return h.hexdigest()


def prepare_policy(p, snapshot, arm):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    policy = TraceableMLXPolicy(str(snapshot), max_prompt_tokens=p["max_prompt_tokens"])
    model = policy.model
    model.set_dtype(mx.float32 if arm == "fp32" else mx.bfloat16)
    if arm == "q4":
        nn.quantize(model, **p["quantization"])
    mx.eval(model.parameters())
    model.freeze()
    adapter = p["adapter"]
    mx.random.seed(adapter["seed"])
    linear_to_lora_layers(
        model, adapter["num_layers"],
        {key: adapter[key] for key in ("rank", "scale", "dropout", "keys")},
    )
    model.freeze()
    model.layers[-1].self_attn.q_proj.unfreeze(
        keys=["lora_a", "lora_b"], recurse=False, strict=True,
    )
    model.eval()
    mx.eval(model.parameters())
    trainable = [name for name, _ in tree_flatten(model.trainable_parameters())]
    if len(trainable) != 2 or not all("lora_" in name for name in trainable):
        raise ValueError(f"unexpected trainable parameter inventory: {trainable}")
    dtype_counts = dict(Counter(str(v.dtype) for _, v in tree_flatten(model.parameters())))
    quantized_modules = sum(
        isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        for _, module in model.named_modules()
    )
    if (arm == "q4") != (quantized_modules > 0):
        raise ValueError("quantized module inventory does not match arm")
    return policy, trainable, dtype_counts, quantized_modules


def collect_group(policy, p, context, checkpoint):
    initial = p["prompt_prefix"] + p["filler"] * context["repetitions"]
    samples = []
    raw = []
    for member, seed in enumerate(p["seeds"]):
        messages = [{"role": "user", "content": initial}]
        segments = []
        for turn in range(2):
            result = policy.generate(
                messages, max_tokens=p["max_new_tokens"], temperature=p["temperature"],
                top_p=p["top_p"], seed=seed + turn * p["second_turn_seed_offset"],
            )
            if (not result.prompt_ids or not result.completion_ids
                    or result.sampling_behavior_logprobs is None
                    or len(result.completion_ids) > p["max_new_tokens"]):
                raise ValueError("incomplete sampled diagnostic action")
            segments.append(G.TrajectorySegment(
                prompt_ids=result.prompt_ids, completion_ids=result.completion_ids,
                behavior_logprobs=result.sampling_behavior_logprobs,
            ))
            raw.append({
                "member": member, "turn": turn,
                "seed": seed + turn * p["second_turn_seed_offset"],
                "prompt_ids": result.prompt_ids,
                "completion_ids": result.completion_ids,
                "sampled_logprobs": result.sampling_behavior_logprobs,
                "finish_reason": result.finish_reason,
            })
            messages = messages + [
                {"role": "assistant", "content": result.text},
                {"role": "user", "content": p["observation"]},
            ]
        first = segments[0]
        samples.append(G.Sample(
            instance_id="diagnostic-" + context["id"],
            behavior_checkpoint_id=checkpoint, behavior_step=0,
            prompt_ids=first.prompt_ids, completion_ids=first.completion_ids,
            behavior_logprobs=list(first.behavior_logprobs or []),
            reward=0.0, verifier="no-task-no-grader", sample_index=member,
            trajectory_segments=segments,
            meta={
                "behavior_logprob_source": "sampling_distribution",
                "behavior_group_size": len(p["seeds"]),
                "behavior_temperature": p["temperature"],
                "behavior_top_p": p["top_p"],
            },
        ))
    return samples, raw


def arithmetic_control(model, samples, p, mx):
    """Separate same-uncached-logits arithmetic from cached-vs-uncached path differences."""
    records = []
    for sample in samples:
        for turn, segment in enumerate(sample.trajectory_segments or []):
            tokens, _, start = G.sample_tensors(segment, 0)
            # Same projection and full-context path as the trainer. Recreate the
            # upstream sampler's native-dtype normalization and temperature scaling.
            logits = G._logits(model, tokens[:, :-1], start)[0]
            upstream = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            uncached_native = [
                _sampling_token_logprob(upstream[i], token, p["temperature"], mx)
                for i, token in enumerate(segment.completion_ids)
            ]
            plain = G.token_logprobs(model, tokens, temperature=p["temperature"], start=start)
            mx.eval(plain)
            plain_scores = [float(x) for x in plain[0].tolist()]
            sampled = segment.behavior_logprobs or []
            if not len(sampled) == len(uncached_native) == len(plain_scores):
                raise ValueError("arithmetic control vector mismatch")
            if not all(math.isfinite(x) for x in plain_scores):
                raise ValueError("nonfinite arithmetic control scores")
            records.append({
                "member": sample.sample_index, "turn": turn,
                "uncached_native_sampler_logprobs": uncached_native,
                "uncached_trainer_logprobs": plain_scores,
                "cached_vs_uncached_native_max": max(
                    abs(a - b) for a, b in zip(sampled, uncached_native)
                ),
                "uncached_native_vs_trainer_max": max(
                    abs(a - b) for a, b in zip(uncached_native, plain_scores)
                ),
            })
    return records


def run(p, snapshot: Path, arm: str, output: Path, device: str):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    if output.exists():
        raise FileExistsError(output)
    if device == "gpu" and not mx.metal.is_available():
        raise RuntimeError("Metal requested but unavailable; refusing CPU fallback")
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("run must be offline after the pinned snapshot is downloaded")
    t0 = time.monotonic()
    files = verify_snapshot(snapshot, p)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise ValueError("diagnostic source tree must be clean")
    report = {
        "schema": "vare.grpo.pretrained_sampling_alignment_result/v1",
        "protocol_sha256": digest(PROTOCOL), "source_commit": source_sha,
        "model": p["model"], "snapshot_files": files, "arm": arm,
        "device": str(mx.default_device()), "requested_device": device,
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "mlx", "mlx-lm", "numpy", "transformers", "huggingface-hub", "safetensors"
        )},
        "optimizer_updated": False, "task_success_measured": False,
        "heldout_accessed": False, "status": "INCOMPLETE",
        "groups": [],
    }
    try:
        policy, trainable, dtypes, quantized = prepare_policy(p, snapshot, arm)
        model = policy.model
        before = parameter_digest(model, mx, tree_flatten)
        report.update(
            parameter_digest_before=before, trainable=trainable,
            dtype_counts=dtypes, quantized_modules=quantized,
        )
        token_total = 0
        for context in p["contexts"]:
            if time.monotonic() - t0 > p["limits"]["maximum_wall_seconds_per_arm"]:
                raise TimeoutError("diagnostic wall budget exceeded")
            samples, raw = collect_group(policy, p, context, before)
            samples_before = copy.deepcopy(samples)
            cfg = G.GRPOConfig(temperature=p["temperature"], group_size=len(p["seeds"]))
            alignment = G.audit_on_policy_sampling_alignment(
                model, samples, cfg, policy_checkpoint_id=before, policy_step=0,
                max_action_segments=p["limits"]["max_action_segments_per_group"],
                max_completion_tokens=p["limits"]["max_completion_tokens_per_group"],
                max_sequence_tokens=p["max_sequence_tokens"],
                tolerance=p["tolerance"],
            )
            control = arithmetic_control(model, samples, p, mx)
            if samples != samples_before:
                raise RuntimeError("diagnostic altered sampled records")
            token_total += alignment["n_completion_tokens"]
            if token_total > p["limits"]["maximum_generated_tokens_per_arm"]:
                raise ValueError("diagnostic token budget exceeded")
            row = {
                "context": context["id"], "alignment": alignment,
                "actions": raw, "arithmetic_control": control,
                "note": "Strict guard hypothetical if these actions were selected into an update; all rewards are dummy zeros.",
            }
            report["groups"].append(row)
            print("VARE_ALIGNMENT_GROUP " + json.dumps({
                "arm": arm, "context": context["id"], "tokens": alignment["n_completion_tokens"],
                "prompt_lengths": [len(action["prompt_ids"]) for action in raw],
                "metrics": alignment["metrics"],
                "would_reject_if_selected": alignment["would_reject_lag_zero_update"],
                "cached_vs_uncached_native_max": max(
                    action["cached_vs_uncached_native_max"] for action in control
                ),
                "uncached_native_vs_trainer_max": max(
                    action["uncached_native_vs_trainer_max"] for action in control
                ),
            }, sort_keys=True, allow_nan=False), flush=True)
        after = parameter_digest(model, mx, tree_flatten)
        report["parameter_digest_after"] = after
        if before != after or model.training:
            raise RuntimeError("model bytes or evaluation mode changed during diagnostic")
        report["parameter_bytes_unchanged"] = True
        report["generated_tokens"] = token_total
        report["status"] = "MEASURED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - t0
        report["process_peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["rss_unit"] = "bytes" if sys.platform == "darwin" else "KiB"
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print("VARE_ALIGNMENT_FINAL " + json.dumps({
            "arm": arm, "status": report["status"],
            "source_commit": source_sha, "protocol_sha256": report["protocol_sha256"],
            "generated_tokens": report.get("generated_tokens"),
            "parameter_bytes_unchanged": report.get("parameter_bytes_unchanged", False),
            "wall_seconds": report["wall_seconds"], "process_peak_rss": report["process_peak_rss"],
            "rss_unit": report["rss_unit"], "error": report.get("error"),
            "result_sha256": digest(output),
        }, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--snapshot-path-file", required=True, type=Path)
    measure = sub.add_parser("run")
    measure.add_argument("--snapshot", required=True, type=Path)
    measure.add_argument("--arm", required=True, choices=("fp32", "bf16", "q4"))
    measure.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    measure.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    p = load_protocol()
    if args.command == "download":
        download(p, args.snapshot_path_file)
    else:
        run(p, args.snapshot, args.arm, args.output, args.device)


if __name__ == "__main__":
    main()
