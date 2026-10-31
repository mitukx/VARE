"""Disk / memory probes, the analytic memory estimate, and stop rules."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Mapping

GiB = 1024 ** 3


class StopRule(RuntimeError):
    """A frozen protocol stop rule fired. The run must halt, not continue."""

    def __init__(self, rule: str, detail: Mapping[str, Any]):
        super().__init__(f"STOP[{rule}]: {dict(detail)}")
        self.rule = rule
        self.detail = dict(detail)


def free_disk_bytes(path: Path) -> int:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def dir_bytes(path: Path) -> int:
    total = 0
    if not Path(path).exists():
        return 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            total += st.st_size
    return total


def memory_snapshot() -> dict[str, Any]:
    """Total/available memory and swap used, in bytes (macOS or Linux)."""
    out: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine()}
    if platform.system() == "Darwin":
        def sysctl(name: str) -> str:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()

        out["total"] = int(sysctl("hw.memsize") or 0)
        out["model"] = sysctl("hw.model")
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", vm).group(1)) if "page size" in vm else 16384
        pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in re.findall(r"^(.+?):\s+(\d+)\.?$", vm, re.M)}
        avail = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
        out["available"] = avail * page
        m = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage"))
        out["swap_used"] = int(float(m.group(1)) * 1024 ** 2) if m else None
    else:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
        out["total"] = info.get("MemTotal")
        out["available"] = info.get("MemAvailable")
        out["swap_used"] = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
    return out


def estimate_lora_training_bytes(*, weights_bytes: int, seq_len: int, vocab: int, hidden: int,
                                 n_layers: int, lora_layers: int, rank: int, logits_bytes_per: int = 4,
                                 act_bytes_per: int = 2) -> dict[str, int]:
    """Coarse upper-bound style estimate for mlx-lm LoRA with grad checkpointing.

    logits: forward logits + their gradient + log-softmax temporary (3 copies).
    checkpointed activations: one hidden-state tensor per layer plus a full
    recompute of one layer (~ 16 * hidden per token, attention via SDPA).
    LoRA params + Adam: tiny (q/k/v/o-style targets, 4 matrices per layer).
    framework overhead: fixed 0.75 GiB.
    """
    logits = 3 * seq_len * vocab * logits_bytes_per
    ckpt = n_layers * seq_len * hidden * act_bytes_per
    recompute = 16 * seq_len * hidden * act_bytes_per
    lora_params = lora_layers * 4 * 2 * rank * hidden
    lora_state = lora_params * 4 * 4  # fp32 weight, grad, adam m, v
    overhead = int(0.75 * GiB)
    total = weights_bytes + logits + ckpt + recompute + lora_state + overhead
    return {"weights": weights_bytes, "logits": logits, "checkpointed_activations": ckpt,
            "layer_recompute": recompute, "lora_and_optimizer": lora_state,
            "overhead": overhead, "total": total}


def check_disk(path: Path, min_free_gib: float, *, stage: str) -> int:
    free = free_disk_bytes(path)
    if free < min_free_gib * GiB:
        raise StopRule("disk_budget", {"stage": stage, "free_gib": round(free / GiB, 2),
                                       "min_free_gib": min_free_gib})
    return free


def check_memory(min_available_gib: float, *, stage: str) -> dict[str, Any]:
    snap = memory_snapshot()
    avail = snap.get("available") or 0
    if avail < min_available_gib * GiB:
        raise StopRule("memory_headroom", {"stage": stage, "available_gib": round(avail / GiB, 2),
                                           "min_available_gib": min_available_gib})
    return snap
