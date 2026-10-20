#!/usr/bin/env python3
"""Run a pinned same-model MLX Metal/CPU decoding microbenchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_backend_microbenchmark_v1.lock.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def system_swap() -> dict[str, int]:
    raw = subprocess.run(
        ["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True
    ).stdout
    import re

    match = re.search(r"used = ([0-9.]+)M.*free = ([0-9.]+)M", raw)
    if match is None:
        raise RuntimeError("cannot parse macOS swap usage")
    return {
        "used_bytes": int(float(match.group(1)) * 1024**2),
        "free_bytes": int(float(match.group(2)) * 1024**2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = args.model_snapshot.resolve()
    if model.name != protocol["model"]["revision"] or not model.is_dir():
        raise RuntimeError("model snapshot revision mismatch")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Apple Silicon macOS is required")
    if importlib.metadata.version("mlx") != protocol["software"]["mlx"]:
        raise RuntimeError("MLX version mismatch")
    if importlib.metadata.version("mlx-lm") != protocol["software"]["mlx-lm"]:
        raise RuntimeError("MLX-LM version mismatch")
    start_swap = system_swap()
    if (
        start_swap["free_bytes"]
        < protocol["hardware_gate"]["abort_if_swap_free_below_bytes"]
    ):
        raise RuntimeError("swap safety gate failed")
    disk_free_start = shutil.disk_usage(output).free
    if disk_free_start < protocol["hardware_gate"]["abort_if_disk_free_below_bytes"]:
        raise RuntimeError("disk safety gate failed")

    prompt = protocol["decoding"]["prompt"]
    raw: dict[str, dict] = {}
    output_text: dict[str, list[str]] = {}
    for backend, device in (("metal", mx.gpu), ("cpu", mx.cpu)):
        mx.set_default_device(device)
        mx.random.seed(101)
        mx.reset_peak_memory()
        load_started = time.perf_counter()
        model_obj, tokenizer = load(str(model))
        load_seconds = time.perf_counter() - load_started
        prompt_tokens = len(tokenizer.encode(prompt))
        generate(
            model_obj,
            tokenizer,
            prompt=prompt,
            max_tokens=protocol["decoding"]["warmup_tokens"],
            verbose=False,
        )
        times: list[float] = []
        outputs: list[str] = []
        token_counts: list[int] = []
        for _ in range(protocol["decoding"]["timed_repetitions_per_backend"]):
            started = time.perf_counter()
            text = generate(
                model_obj,
                tokenizer,
                prompt=prompt,
                max_tokens=protocol["decoding"]["max_new_tokens"],
                verbose=False,
            )
            times.append(time.perf_counter() - started)
            outputs.append(text)
            token_counts.append(len(tokenizer.encode(text)))
        rates = [
            count / seconds for count, seconds in zip(token_counts, times, strict=True)
        ]
        raw[backend] = {
            "selected_device": str(mx.default_device()),
            "load_seconds": round(load_seconds, 6),
            "prompt_tokens": prompt_tokens,
            "output_tokens": token_counts,
            "inference_seconds": [round(value, 6) for value in times],
            "tokens_per_second": [round(value, 6) for value in rates],
            "median_tokens_per_second": round(sorted(rates)[len(rates) // 2], 6),
            "mlx_peak_memory_bytes": int(mx.get_peak_memory())
            if backend == "metal"
            else None,
            "output_sha256": [sha256(value.encode()) for value in outputs],
        }
        output_text[backend] = outputs
        del model_obj, tokenizer
        import gc

        gc.collect()
        mx.clear_cache()

    summary = {
        "protocol_id": protocol["protocol_id"],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "metal_available": mx.metal.is_available(),
        },
        "model": {
            "repository": protocol["model"]["repository"],
            "revision": protocol["model"]["revision"],
            "license": protocol["model"]["license"],
            "files": tree_hashes(model),
        },
        "results": raw,
        "same_output_each_backend": all(
            len(set(items)) == 1 for items in output_text.values()
        ),
        "same_output_between_backends": len(
            set(output_text["metal"] + output_text["cpu"])
        )
        == 1,
        "resources": {
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "swap_start": start_swap,
            "swap_end": system_swap(),
            "disk_free_start_bytes": disk_free_start,
            "disk_free_end_bytes": shutil.disk_usage(output).free,
        },
        "claim_limit": protocol["interpretation_limit"],
    }
    (output / "raw_outputs.json").write_text(
        json.dumps(output_text, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    files = {
        path.name: sha256(path.read_bytes())
        for path in output.iterdir()
        if path.is_file()
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": sha256(PROTOCOL.read_bytes()),
                "runner_source_sha256": sha256(Path(__file__).read_bytes()),
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "metal_tps": raw["metal"]["median_tokens_per_second"],
                "cpu_tps": raw["cpu"]["median_tokens_per_second"],
                "same_output": summary["same_output_between_backends"],
                "metal_peak_bytes": raw["metal"]["mlx_peak_memory_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["same_output_between_backends"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
