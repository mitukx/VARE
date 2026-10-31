"""LoRA training arms on top of mlx-lm.

Arms (all share model, LoRA config, optimizer, iterations, sequence limit,
seed; only the training data differ):

* ``no_update`` - the frozen base policy; no training.
* ``sft_reference`` - SFT on reference-fix demonstrations rendered in the
  policy's edit format. Each demonstration is admitted only if the trusted
  grader verified it during calibration. These are dataset reference fixes,
  not student trajectories, and their tasks are Train-only.
* ``rft_self`` - rejection-sampling fine-tuning (STaR/RFT style): SFT on the
  base policy's own sampled outputs on Train tasks that the trusted grader
  accepted.

GRPO is a stub (see ``grpo.py``); it is not part of the v1 comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from .edit_format import reference_fix_blocks, render_blocks
from .prompting import build_messages


def sft_examples(tasks: Iterable[Mapping[str, Any]], buggy: Mapping[str, str],
                 calibration: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tasks:
        cal = calibration.get(t["instance_id"])
        if not cal or not cal.get("qualified"):
            continue
        _, blocks = reference_fix_blocks(t["_patch"])
        msgs = build_messages(t, buggy[t["instance_id"]])
        out.append({"messages": msgs + [{"role": "assistant", "content": render_blocks(blocks)}],
                    "instance_id": t["instance_id"], "task_group": t["upstream_repo"],
                    "origin": "reference_fix_verified"})
    return out


def rft_examples(tasks: Iterable[Mapping[str, Any]], buggy: Mapping[str, str],
                 rollouts: Iterable[Mapping[str, Any]], *, max_per_task: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for r in rollouts:
        if r.get("reward") == 1:
            by_task.setdefault(r["instance_id"], []).append(r)
    out = []
    for t in tasks:
        seen: set[str] = set()
        for r in sorted(by_task.get(t["instance_id"], []), key=lambda r: r["sample_index"]):
            if r["output_sha256"] in seen or len(seen) >= max_per_task:
                continue
            seen.add(r["output_sha256"])
            msgs = build_messages(t, buggy[t["instance_id"]])
            out.append({"messages": msgs + [{"role": "assistant", "content": r["output_text"]}],
                        "instance_id": t["instance_id"], "task_group": t["upstream_repo"],
                        "origin": "base_policy_sample_trusted_pass",
                        "source_checkpoint_id": r.get("checkpoint_id"), "source_group_id": r.get("group_id"),
                        "source_sample_index": r["sample_index"], "source_output_sha256": r["output_sha256"]})
    return out


def write_dataset(examples: list[dict[str, Any]], out_dir: Path, *, valid_count: int, seed: int) -> dict[str, Any]:
    """Write mlx-lm chat-format train/valid jsonl. Valid rows are Train-task
    rows held aside for loss monitoring only (never the held-out split)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    valid = rows[:valid_count] if len(rows) > valid_count else rows[:1]
    train = rows[len(valid):] if len(rows) > valid_count else rows
    for name, part in (("train", train), ("valid", valid)):
        with (out_dir / f"{name}.jsonl").open("w") as fh:
            for r in part:
                fh.write(json.dumps({"messages": r["messages"]}) + "\n")
        # Provenance sidecar (same row order) so every training row maps back
        # to its task, task group and, for RFT, the exact rollout it came from.
        with (out_dir / f"{name}.provenance.jsonl").open("w") as fh:
            for i, r in enumerate(part):
                fh.write(json.dumps({"row": i, **{k: v for k, v in r.items() if k != "messages"}},
                                    sort_keys=True) + "\n")
    return {"train_rows": len(train), "valid_rows": len(valid),
            "train_tasks": len({r["instance_id"] for r in train})}


def lora_config(*, model: str, data_dir: Path, adapter_dir: Path, hp: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "optimizer": "adam",
        "mask_prompt": True,
        "num_layers": hp["num_layers"],
        "batch_size": hp["batch_size"],
        "grad_accumulation_steps": hp.get("grad_accumulation_steps", 1),
        "iters": hp["iters"],
        "val_batches": hp["val_batches"],
        "learning_rate": hp["learning_rate"],
        "steps_per_report": hp["steps_per_report"],
        "steps_per_eval": hp["steps_per_eval"],
        "save_every": hp["save_every"],
        "adapter_path": str(adapter_dir),
        "max_seq_length": hp["max_seq_length"],
        "grad_checkpoint": True,
        "seed": hp["seed"],
        "lora_parameters": {"rank": hp["lora_rank"], "scale": hp["lora_scale"], "dropout": 0.0},
    }


def run_lora(cfg: Mapping[str, Any], log_path: Path, *, timeout_s: float,
             python: str = sys.executable, resume: bool = True) -> dict[str, Any]:
    """Run mlx-lm LoRA training in a subprocess (``lora_worker``). Resumes from
    the newest saved adapter checkpoint after a crash."""
    adapter_dir = Path(cfg["adapter_path"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    final = adapter_dir / "adapters.safetensors"
    done_marker = adapter_dir / "TRAINING_COMPLETE.json"
    if done_marker.exists():
        return json.loads(done_marker.read_text())
    cfg = dict(cfg)
    ckpts = sorted(adapter_dir.glob("0*_adapters.safetensors"))
    resumed_from = None
    if resume and ckpts:
        resumed_from = str(ckpts[-1])
        done_iters = int(ckpts[-1].name.split("_", 1)[0])
        cfg["resume_adapter_file"] = resumed_from
        cfg["iters"] = max(1, int(cfg["iters"]) - done_iters)
    cfg_path = adapter_dir / "lora_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    metrics_path = adapter_dir / "metrics.jsonl"
    t0 = time.time()
    with log_path.open("a") as log:
        proc = subprocess.run([python, "-m", "vare.learning_pipeline.lora_worker", str(cfg_path), str(metrics_path)],
                              stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
    metrics = read_metrics(metrics_path)
    result = {"returncode": proc.returncode, "seconds": round(time.time() - t0, 1),
              "resumed_from": resumed_from, "config_path": str(cfg_path), "log_path": str(log_path),
              "peak_memory_gb": max((m.get("peak_memory", 0.0) for m in metrics if m["kind"] == "train"), default=None),
              "trained_tokens": max((m.get("trained_tokens", 0) for m in metrics if m["kind"] == "train"), default=None),
              "final_train_loss": next((m["train_loss"] for m in reversed(metrics) if m["kind"] == "train"), None),
              "val_losses": [m.get("val_loss") for m in metrics if m["kind"] == "valid"]}
    if proc.returncode != 0 or not final.exists():
        raise RuntimeError(f"mlx_lm lora failed: {result}")
    done_marker.write_text(json.dumps(result, indent=2))
    return result


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def adapter_fingerprint(adapter_dir: Path) -> dict[str, Any]:
    """sha256 of the saved adapter plus nonzero-delta check on LoRA B matrices
    (B is zero-initialized, so any nonzero B proves the update happened)."""
    import hashlib

    path = adapter_dir / "adapters.safetensors"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    info: dict[str, Any] = {"adapters_sha256": digest, "bytes": path.stat().st_size}
    try:
        import mlx.core as mx

        weights = mx.load(str(path))
        b_keys = [k for k in weights if k.endswith("lora_b")]
        info["lora_b_tensors"] = len(b_keys)
        info["lora_b_nonzero_tensors"] = sum(int(mx.any(weights[k] != 0).item()) for k in b_keys)
        info["lora_b_max_abs"] = max((float(mx.abs(weights[k]).max().item()) for k in b_keys), default=0.0)
    except Exception as exc:  # noqa: BLE001
        info["inspect_error"] = repr(exc)
    return info
