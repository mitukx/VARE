#!/usr/bin/env python3
"""Isolate Metal attention and projection path differences on fixed tokens."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import diagnose_grpo_backend_alignment as B  # noqa: E402
from scripts import diagnose_grpo_pretrained_sampling as D  # noqa: E402
from vare import grpo_mlx as G  # noqa: E402

PROTOCOL = ROOT / "protocols/grpo_metal_path_isolation_v4.json"
PRIOR = ROOT / "protocols/grpo_pretrained_sampling_alignment_v3.json"


def protocol():
    p = json.loads(PROTOCOL.read_text())
    if (p["schema"] != "vare.grpo.metal_path_isolation/v4"
            or p["prior_protocol_sha256"] != D.digest(PRIOR)
            or p["tolerance"] != G.ON_POLICY_LOG_RATIO_TOLERANCE):
        raise ValueError("invalid path-isolation protocol or changed prior protocol")
    return p


def explicit_attention(q, k, v, *, cache=None, scale, mask=None, sinks=None):
    """Transparent GQA reference; only ordinary, unquantized KV cache."""
    import mlx.core as mx

    if sinks is not None or hasattr(cache, "bits"):
        raise ValueError("reference attention does not support sinks or quantized KV cache")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("unexpected attention shapes")
    if q.shape[1] % k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible grouped-query attention")
    ratio = q.shape[1] // k.shape[1]
    keys = mx.repeat(k.astype(mx.float32), ratio, axis=1)
    values = mx.repeat(v.astype(mx.float32), ratio, axis=1)
    scores = (q.astype(mx.float32) @ keys.transpose(0, 1, 3, 2)) * scale
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("unsupported attention mask")
        q_len, k_len = q.shape[-2], k.shape[-2]
        mask = mx.arange(k_len)[None, :] <= (mx.arange(q_len)[:, None] + k_len - q_len)
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -float("inf"))
        else:
            scores = scores + mask.astype(mx.float32)
    return (mx.softmax(scores, axis=-1, precise=True) @ values).astype(q.dtype)


@contextmanager
def attention_mode(use_explicit):
    from mlx_lm.models import llama

    original = llama.scaled_dot_product_attention
    if use_explicit:
        llama.scaled_dot_product_attention = explicit_attention
    try:
        yield
    finally:
        llama.scaled_dot_product_attention = original


def native_logprob(logits, target, temperature):
    import mlx.core as mx

    native = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    scaled = (native * (1.0 / temperature)).astype(mx.float32)
    return scaled[..., target] - mx.logsumexp(scaled, axis=-1)


def maximum_difference(a, b):
    import mlx.core as mx

    value = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(value)
    value = float(value.item())
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite path difference")
    return value


def gradient_summary(gradients):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = list(tree_flatten(gradients))
    if not flat:
        raise ValueError("empty gradient")
    finite = all(bool(mx.all(mx.isfinite(g)).item()) for _, g in flat)
    norm = math.sqrt(sum(float(mx.sum(g.astype(mx.float32) ** 2).item()) for _, g in flat))
    if not finite or not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError("expected a finite nonzero gradient")
    return {"finite": finite, "l2": norm, "parameters": len(flat)}


def attention_probe(p, precision, length):
    import mlx.core as mx
    import mlx.nn as nn

    fixture = p["attention_fixture"]
    dtype = mx.float32 if precision == "fp32" else mx.bfloat16
    mx.random.seed(fixture["seed"] + length)
    q = mx.random.normal((1, fixture["heads"], length, fixture["head_dim"])).astype(dtype)
    k = mx.random.normal((1, fixture["kv_heads"], length, fixture["head_dim"])).astype(dtype)
    v = mx.random.normal(k.shape).astype(dtype)
    mx.eval(q, k, v)
    scale = fixture["head_dim"] ** -0.5

    class Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = q

        def __call__(self, explicit):
            if explicit:
                return explicit_attention(self.q, k, v, scale=scale, mask="causal")
            return mx.fast.scaled_dot_product_attention(self.q, k, v, scale=scale, mask="causal")

    module = Probe()
    results = {}
    for name in ("stock", "explicit"):
        explicit = name == "explicit"
        plain = module(explicit)
        mx.eval(plain)

        def objective():
            out = module(explicit)
            return mx.mean(out.astype(mx.float32) ** 2), out

        (_, traced), grad = nn.value_and_grad(module, objective)()
        mx.eval(traced, grad)
        results[name] = {
            "plain_vs_differentiated_max": maximum_difference(plain, traced),
            "gradient": gradient_summary(grad),
        }
    results["stock_vs_explicit_plain_max"] = maximum_difference(module(False), module(True))
    return {"length": length, "precision": precision, "results": results}


def model_probe(model, sequence, p, mode):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache

    ids = mx.array([sequence["prompt_ids"]])
    target = sequence["target_id"]
    start = len(sequence["prompt_ids"]) - 1
    explicit = mode.startswith("explicit_")
    full = mode.endswith("_full")

    def logits():
        if full:
            return model(ids)[:, -1:]
        return G._logits(model, ids, start)

    def score():
        values = logits()
        return values, native_logprob(values, target, p["temperature"])

    with attention_mode(explicit):
        plain_logits, plain_lp = score()
        mx.eval(plain_logits, plain_lp)
        cache = make_prompt_cache(model)
        cached_logits = model(ids, cache=cache)[:, -1:]
        cached_lp = native_logprob(cached_logits, target, p["temperature"])
        mx.eval(cached_logits, cached_lp)

        def objective():
            values, lp = score()
            return -lp.mean(), (values, lp)

        (_, (traced_logits, traced_lp)), grad = nn.value_and_grad(model, objective)()
        mx.eval(traced_logits, traced_lp, grad)
        values = {
            "plain_logprob": float(plain_lp.item()),
            "cached_prefill_logprob": float(cached_lp.item()),
            "differentiated_logprob": float(traced_lp.item()),
            "plain_vs_differentiated_max": maximum_difference(plain_lp, traced_lp),
            "cached_prefill_vs_differentiated_max": maximum_difference(cached_lp, traced_lp),
            "plain_vs_differentiated_logits_max": maximum_difference(plain_logits, traced_logits),
            "cached_prefill_vs_differentiated_logits_max": maximum_difference(cached_logits, traced_logits),
            "gradient": gradient_summary(grad),
        }
        if not all(math.isfinite(values[k]) for k in (
            "plain_logprob", "cached_prefill_logprob", "differentiated_logprob",
        )):
            raise FloatingPointError("nonfinite score")
        values["within_tolerance"] = (
            max(values["plain_vs_differentiated_max"],
                values["cached_prefill_vs_differentiated_max"]) <= p["tolerance"]
        )
        return {"mode": mode, "context": sequence["id"], "prompt_length": start + 1,
                "target_id": target, **values}


def preflight(p):
    import mlx.core as mx

    return B.select_backend(p["backend"], mx, system=platform.system(), machine=platform.machine())


def run(p, precision, snapshot, output):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    if output.exists():
        raise FileExistsError(output)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("measurement must be offline")
    host = preflight(p)
    if host["status"] != "READY":
        raise RuntimeError("registered Metal backend unavailable")
    started = time.monotonic()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True,
    ).strip():
        raise ValueError("source tree must be clean")
    record = {
        "schema": "vare.grpo.metal_path_isolation_result/v4",
        "source_commit": head, "protocol_sha256": D.digest(PROTOCOL), "precision": precision,
        "backend": host, "platform": platform.platform(), "python": platform.python_version(),
        "versions": {n: importlib.metadata.version(n) for n in (
            "mlx", "mlx-lm", "numpy", "transformers", "huggingface-hub", "safetensors",
        )},
        "status": "INCOMPLETE", "optimizer_updated": False, "model_generated_tokens": 0,
        "task_success_measured": False, "heldout_accessed": False, "attention": [], "paths": [],
    }
    try:
        for length in p["attention_fixture"]["lengths"]:
            row = attention_probe(p, precision, length)
            record["attention"].append(row)
            print("VARE_METAL_ATTENTION " + json.dumps(row, sort_keys=True), flush=True)
        record["snapshot_files"] = D.verify_snapshot(snapshot, p)
        policy, trainable, dtypes, quantized = D.prepare_policy(p, snapshot, precision)
        model = policy.model
        if not isinstance(model, llama.Model) or quantized:
            raise ValueError("only the pinned unquantized Llama implementation is registered")
        before = D.parameter_digest(model, mx, tree_flatten)
        record.update(parameter_digest_before=before, dtype_counts=dtypes, trainable=trainable)
        original = llama.scaled_dot_product_attention
        for sequence in p["sequences"]:
            for mode in p["modes"]:
                if time.monotonic() - started > p["limits"]["maximum_wall_seconds_per_arm"]:
                    raise TimeoutError("measurement time limit")
                row = model_probe(model, sequence, p, mode)
                if llama.scaled_dot_product_attention is not original:
                    raise RuntimeError("attention intervention escaped diagnostic scope")
                record["paths"].append(row)
                print("VARE_METAL_PATH " + json.dumps(row, sort_keys=True), flush=True)
        if len(record["paths"]) != p["limits"]["maximum_model_path_measurements_per_arm"]:
            raise ValueError("incomplete path inventory")
        after = D.parameter_digest(model, mx, tree_flatten)
        if before != after or model.training:
            raise RuntimeError("parameters or evaluation mode changed")
        record["parameter_digest_after"] = after
        record["parameter_bytes_unchanged"] = True
        record["passing_paths"] = [
            mode for mode in p["modes"]
            if all(row["within_tolerance"] for row in record["paths"] if row["mode"] == mode)
        ]
        # Report cross-mode change in the plain score, not semantic equivalence.
        record["explicit_vs_stock_plain_delta"] = [
            {"context": sequence["id"], "projection": projection,
             "abs_delta": abs(
                next(r["plain_logprob"] for r in record["paths"]
                     if r["context"] == sequence["id"] and r["mode"] == f"explicit_{projection}")
                - next(r["plain_logprob"] for r in record["paths"]
                       if r["context"] == sequence["id"] and r["mode"] == f"stock_{projection}")
             )}
            for sequence in p["sequences"] for projection in ("selected", "full")
        ]
        record["status"] = "MEASURED"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        B.write_json(output, record)
        print("VARE_METAL_PATH_FINAL " + json.dumps({
            "precision": precision, "status": record["status"],
            "source_commit": head, "protocol_sha256": record["protocol_sha256"],
            "result_sha256": D.digest(output), "passing_paths": record.get("passing_paths"),
            "parameter_bytes_unchanged": record.get("parameter_bytes_unchanged", False),
            "wall_seconds": record["wall_seconds"], "error": record.get("error"),
        }, sort_keys=True), flush=True)


def main():
    p = protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("preflight")
    probe.add_argument("--output", type=Path, required=True)
    download = sub.add_parser("download")
    download.add_argument("--snapshot-path-file", type=Path, required=True)
    measure = sub.add_parser("run")
    measure.add_argument("--precision", choices=p["arms"], required=True)
    measure.add_argument("--snapshot", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(p)
        B.write_json(args.output, result)
        print("VARE_METAL_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    elif args.command == "download":
        D.download(p, args.snapshot_path_file)
    else:
        run(p, args.precision, args.snapshot, args.output)


if __name__ == "__main__":
    main()
