"""Measure whether the frozen Qwen3-4B Development study can safely start."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any

MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
EXPECTED_WEIGHT_SHA256 = "2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f"
MIN_AVAILABLE_BYTES = 4 * 1024**3
MIN_FREE_SWAP_BYTES = 512 * 1024**2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _swap() -> dict[str, int]:
    raw = subprocess.run(["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True, timeout=10).stdout
    result: dict[str, int] = {}
    for name in ("total", "used", "free"):
        start = raw.index(f"{name} = ") + len(f"{name} = ")
        end = raw.index("M", start)
        result[f"{name}_bytes"] = int(float(raw[start:end]) * 1024**2)
    return result


def _available_memory_bytes() -> int:
    """Use native macOS counters so the audit works in the frozen MLX lock."""
    raw = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True, timeout=10).stdout
    page_match = re.search(r"page size of (\d+) bytes", raw)
    if page_match is None:
        raise RuntimeError("vm_stat page size is unavailable")
    page_size = int(page_match.group(1))
    counts: dict[str, int] = {}
    for match in re.finditer(r"Pages (free|inactive|speculative):\s+([\d.]+)", raw):
        counts[match.group(1)] = int(match.group(2).replace(".", ""))
    if not all(name in counts for name in ("free", "inactive", "speculative")):
        raise RuntimeError("vm_stat available page counters are missing")
    return sum(counts.values()) * page_size


def audit(model_snapshot: Path) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("resource gate requires Apple Silicon macOS")
    root = model_snapshot.resolve(strict=True)
    if root.name != MODEL_REVISION:
        raise RuntimeError("local model snapshot revision differs from the pinned Qwen3-4B revision")
    weight_hash = _sha256(root / "model.safetensors")
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("local model checkpoint hash differs from the frozen model identity")
    import mlx.core as mx

    if not mx.metal.is_available() or str(mx.default_device()) != "Device(gpu, 0)":
        raise RuntimeError("MLX Metal GPU is unavailable or is not the default device")
    available_memory = _available_memory_bytes()
    swap = _swap()
    checks = {
        "model_revision_pinned": True,
        "model_weight_hash_pinned": True,
        "mlx_metal_gpu_active": True,
        "available_memory_above_floor": available_memory >= MIN_AVAILABLE_BYTES,
        "free_swap_above_floor": swap["free_bytes"] >= MIN_FREE_SWAP_BYTES,
    }
    return {
        "schema_version": 1,
        "study": "vare-qwen3-4b-coding-agent-development-v6",
        "decision": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": {
            "minimum_available_memory_bytes": MIN_AVAILABLE_BYTES,
            "minimum_free_swap_bytes": MIN_FREE_SWAP_BYTES,
        },
        "model": {
            "repository": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            "revision": root.name,
            "model_safetensors_sha256": weight_hash,
        },
        "runtime": {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "default_device": str(mx.default_device()),
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], check=True, capture_output=True, text=True).stdout.strip(),
            "unified_memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], check=True, capture_output=True, text=True).stdout.strip()),
            "available_memory_bytes": available_memory,
            "swap": swap,
        },
        "rollouts_started": 0,
        "policy_updates": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite resource-gate evidence")
    result = audit(args.model_snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
