#!/usr/bin/env python3
"""Run the frozen ten-update MLX QLoRA feasibility smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_lora_update_smoke_v1.lock.json"
DATA = ROOT / "benchmarks/dev_sft_update_smoke_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def completed_training_steps(log: str) -> list[int]:
    """Read MLX-LM's terminal progress table, including ANSI/CR output."""
    plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", log).replace("\r", "\n")
    return [int(match) for match in re.findall(r"^\s*(\d+)\s+[-+0-9.]", plain, re.M)]


def swap_bytes() -> dict[str, int]:
    output = subprocess.run(
        ["sysctl", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found = re.search(r"used = ([0-9.]+)M.*free = ([0-9.]+)M", output)
    if found is None:
        raise RuntimeError("cannot parse macOS vm.swapusage")
    return {
        "used_bytes": int(float(found.group(1)) * 1024**2),
        "free_bytes": int(float(found.group(2)) * 1024**2),
    }


def run(argv: list[str], *, log: Path, env: dict[str, str]) -> float:
    started = time.perf_counter()
    with log.open("wb") as output:
        subprocess.run(
            argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True
        )
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--converted-model-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model = args.model_snapshot.resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    if model.name != protocol["model"]["revision"]:
        raise RuntimeError(
            "model snapshot directory does not match the frozen revision"
        )
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("this protocol requires Apple Silicon macOS")
    if args.python.resolve() != Path(sys.executable).resolve():
        raise RuntimeError("invoke this script with the pinned MLX Python environment")

    import mlx.core as mx
    from mlx_lm import generate, load
    from safetensors.numpy import load_file
    import importlib.metadata as metadata

    device = str(mx.default_device())
    if device != "Device(gpu, 0)" or not mx.metal.is_available():
        raise RuntimeError(f"Apple Metal is unavailable: {device}")
    if metadata.version("mlx") != protocol["software"]["mlx"]:
        raise RuntimeError("MLX version drift")
    if metadata.version("mlx-lm") != protocol["software"]["mlx-lm"]:
        raise RuntimeError("MLX-LM version drift")
    row = json.loads((DATA / "train.jsonl").read_text(encoding="utf-8"))
    if sha256(DATA / "train.jsonl") != protocol["data"]["sha256"]:
        raise RuntimeError("SFT data hash drift")
    if len(row["messages"]) != 2 or row["messages"][-1]["role"] != "assistant":
        raise RuntimeError("expected one prompt/completion SFT example")

    disk = shutil.disk_usage(out)
    start_swap = swap_bytes()
    if (
        start_swap["free_bytes"]
        < protocol["hardware"]["abort_if_swap_free_below_bytes"]
    ):
        raise RuntimeError("swap-space preflight failed")
    if disk.free < protocol["hardware"]["disk_free_min_bytes"]:
        raise RuntimeError("disk-space preflight failed")
    model_files = hash_tree(model)
    model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if model_config.get("_commit_hash") not in (None, protocol["model"]["revision"]):
        raise RuntimeError("model revision metadata drift")

    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    converted = (
        args.converted_model_output.resolve()
        if args.converted_model_output
        else ROOT / "work" / "mlx-lora-models" / out.name / "model-4bit"
    )
    converted.parent.mkdir(parents=True, exist_ok=True)
    if converted.exists():
        raise FileExistsError(f"refusing to overwrite converted model: {converted}")
    conversion_s = run(
        [
            str(args.python),
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            str(model),
            "--mlx-path",
            str(converted),
            "--quantize",
            "--q-bits",
            "4",
            "--q-group-size",
            "64",
        ],
        log=out / "conversion.log",
        env=env,
    )

    adapter = out / "adapter"
    mx.reset_peak_memory()
    import mlx_lm.lora as lora

    argv = [
        "mlx_lm.lora",
        "--model",
        str(converted),
        "--train",
        "--data",
        str(DATA),
        "--fine-tune-type",
        "lora",
        "--mask-prompt",
        "--num-layers",
        "2",
        "--batch-size",
        "1",
        "--iters",
        "10",
        "--learning-rate",
        "0.00001",
        "--steps-per-report",
        "1",
        "--steps-per-eval",
        "10",
        "--val-batches",
        "0",
        "--adapter-path",
        str(adapter),
        "--save-every",
        "10",
        "--max-seq-length",
        "1024",
        "--grad-checkpoint",
        "--seed",
        "101",
    ]
    saved_argv = sys.argv
    started = time.perf_counter()
    try:
        sys.argv = argv
        with (out / "training.log").open("w", encoding="utf-8") as log:
            from contextlib import redirect_stdout, redirect_stderr

            with redirect_stdout(log), redirect_stderr(log):
                lora.main()
    finally:
        sys.argv = saved_argv
    training_s = time.perf_counter() - started
    mx.eval(mx.zeros((1,)))
    peak_mlx = int(mx.get_peak_memory())
    adapter_files = hash_tree(adapter)
    tensors: dict[str, Any] = {}
    for path in sorted(adapter.glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    nonzero = sum(int((value != 0).sum()) for value in tensors.values())
    learned_nonzero = sum(
        int((value != 0).sum())
        for name, value in tensors.items()
        if name.endswith(".lora_b")
    )
    total = sum(int(value.size) for value in tensors.values())
    if not adapter_files or nonzero == 0:
        raise RuntimeError("adapter is missing or unchanged")
    if learned_nonzero == 0:
        raise RuntimeError("LoRA B matrices have no learned values")

    # Re-open the saved adapter as a new model instance and run deterministic inference.
    mx.clear_cache()
    reloaded, tokenizer = load(str(converted), adapter_path=str(adapter))
    prompt = tokenizer.apply_chat_template(
        row["messages"][:1],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = len(tokenizer.encode(prompt))
    started = time.perf_counter()
    generated = generate(
        reloaded, tokenizer, prompt=prompt, max_tokens=32, verbose=False
    )
    inference_s = time.perf_counter() - started
    output_tokens = len(tokenizer.encode(generated))
    peak_mlx = max(peak_mlx, int(mx.get_peak_memory()))
    end_swap = swap_bytes()
    rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    step_ids = completed_training_steps(
        (out / "training.log").read_text(encoding="utf-8")
    )
    checks = {
        "metal_selected": device == "Device(gpu, 0)" and mx.metal.is_available(),
        "ten_steps_logged": step_ids == list(range(1, 11)),
        "adapter_files_present": bool(adapter_files),
        "adapter_has_learned_lora_b_parameters": learned_nonzero > 0,
        "adapter_reload_and_generation": bool(generated) and output_tokens > 0,
        "peak_mlx_below_limit": peak_mlx < protocol["hardware"]["maximum_memory_bytes"],
        "rss_below_limit": rss_bytes < 8_000_000_000,
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "status": "pass" if all(checks.values()) else "non_pass",
        "checks": checks,
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
            "python": platform.python_version(),
            "mlx": metadata.version("mlx"),
            "mlx_lm": metadata.version("mlx-lm"),
            "device": device,
            "metal_available": mx.metal.is_available(),
        },
        "model": {
            "repository": protocol["model"]["repository"],
            "revision": protocol["model"]["revision"],
            "license": "Apache-2.0",
            "quantization": "4-bit affine, group-size 64",
            "source_files": model_files,
            "converted_files": hash_tree(converted),
        },
        "dataset_sha256": sha256(DATA / "train.jsonl"),
        "training": {
            "objective": "SFT feasibility smoke",
            "configured_updates": 10,
            "learning_rate": 0.00001,
            "batch_size": 1,
            "trainable_layers": 2,
            "training_seconds": round(training_s, 3),
            "conversion_seconds": round(conversion_s, 3),
            "adapter_files": adapter_files,
            "adapter_tensor_count": len(tensors),
            "logged_step_ids": step_ids,
            "adapter_nonzero_parameters": nonzero,
            "adapter_learned_lora_b_parameters": learned_nonzero,
            "adapter_parameters": total,
        },
        "reload_inference": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "seconds": round(inference_s, 3),
            "tokens_per_second": round(output_tokens / inference_s, 3),
            "output": generated,
        },
        "resources": {
            "peak_mlx_bytes": peak_mlx,
            "max_rss_bytes": rss_bytes,
            "swap_start": start_swap,
            "swap_end": end_swap,
            "disk_free_start_bytes": disk.free,
            "disk_free_end_bytes": shutil.disk_usage(out).free,
        },
        "claim_limit": "One training example from the development split; no held-out capability evaluation or optimizer-state resume.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (out / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": sha256(PROTOCOL),
                "dataset_sha256": sha256(DATA / "train.jsonl"),
                "files": hash_tree(out),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "checks": checks,
                "peak_mlx_bytes": peak_mlx,
                "training_seconds": round(training_s, 3),
                "tokens_per_second": round(output_tokens / inference_s, 3),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
