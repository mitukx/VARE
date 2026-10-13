#!/usr/bin/env python3
"""Run one exact-integer update, then save, reload, and log resources.

The held-out 64-task pack is not a training example and is not scored.
mlx and transformers are separate backends. Neither result is a replicate
of the other. There is no flag that changes the step count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.update_cost_smoke import (  # noqa: E402
    PROTOCOL_SHA256,
    V2_MANIFEST_SHA256,
    accept_smoke_record,
    assert_train_example_is_outside_the_heldout_pack,
    validate_protocol,
)

PROTOCOL_PATH = ROOT / "protocols/heldout_exact_integer_update_cost_smoke_v1.json"
PACK_PATH = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
V2_MANIFEST = ROOT / "results/heldout-exact-integer-base-gate-v2/m1-mlx-89f671c/manifest.json"


def _protocol() -> dict[str, Any]:
    blob = PROTOCOL_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PROTOCOL_SHA256:
        raise SystemExit("smoke protocol bytes do not match the frozen hash")
    protocol = json.loads(blob.decode("utf-8"))
    validate_protocol(protocol)
    digest = hashlib.sha256(V2_MANIFEST.read_bytes()).hexdigest()
    if digest != V2_MANIFEST_SHA256:
        raise SystemExit("v2 manifest bytes do not match the smoke pin")
    return protocol


def _code_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _failure(backend: str, exc: BaseException) -> dict[str, Any]:
    return {
        "backend": backend,
        "error": "".join(traceback.format_exception(exc)),
        "heldout_pack_used_for_update": False,
        "loss": None,
        "optimizer_steps": 0,
        "reload_matches_saved": False,
        "resources": {"peak_memory_bytes": None, "wall_seconds": None},
        "trainable_parameters_changed": False,
    }


def _environment(backend: str) -> dict[str, object]:
    record: dict[str, object] = {
        "backend": backend,
        "python": platform.python_version(),
    }
    if backend == "mlx":
        import mlx_lm  # type: ignore[import-not-found]

        record["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    else:
        import peft  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]

        record["cuda"] = bool(torch.cuda.is_available())
        record["peft"] = getattr(peft, "__version__", "unknown")
        record["torch"] = torch.__version__
        record["transformers"] = transformers.__version__
    return record


def _trainable_copy(model: Any) -> dict[str, Any]:
    import mlx.core as mx  # type: ignore[import-not-found]
    from mlx.utils import tree_flatten  # type: ignore[import-not-found]

    copied = {name: mx.array(array) for name, array in tree_flatten(model.trainable_parameters())}
    if copied:
        mx.eval(*copied.values())
    return copied


def _same_leaves(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import mlx.core as mx  # type: ignore[import-not-found]

    if left.keys() != right.keys():
        return False
    for name in left:
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            return False
        if not bool(mx.all(left[name] == right[name]).item()):
            return False
    return True


def _write_mlx_adapter_config(adapter_dir: Path, protocol: dict[str, Any]) -> None:
    payload = {
        "fine_tune_type": "lora",
        "lora_parameters": {
            "dropout": float(protocol["lora"]["dropout"]),
            "keys": list(protocol["lora"]["keys"]),
            "rank": int(protocol["lora"]["rank"]),
            "scale": float(protocol["lora"]["scale"]),
        },
        "num_layers": int(protocol["num_layers"]),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (adapter_dir / "adapter_config.json").write_text(text, encoding="utf-8")


def _mlx(protocol: dict[str, Any], example: dict[str, object], output: Path) -> dict[str, Any]:
    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx.optimizers as optim  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from mlx_lm.tuner.datasets import CacheDataset, CompletionsDataset  # type: ignore[import-not-found]
    from mlx_lm.tuner.trainer import TrainingArgs, train  # type: ignore[import-not-found]
    from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters  # type: ignore[import-not-found]
    from mlx_lm.utils import load  # type: ignore[import-not-found]

    class Capture:
        def __init__(self) -> None:
            self.loss: float | None = None
            self.peak_memory_gb: float | None = None

        def on_train_loss_report(self, info: dict[str, Any]) -> None:
            self.loss = float(info["train_loss"])
            self.peak_memory_gb = float(info["peak_memory"])

    spec = protocol["models"]["mlx"]
    if int(protocol["num_layers"]) != 1 or int(protocol["optimizer_steps"]) != 1:
        raise RuntimeError("mlx smoke refuses to change the frozen step count")
    local = snapshot_download(spec["repo_id"], revision=spec["revision"])
    started = time.perf_counter()
    model, tokenizer = load(local)
    mx.random.seed(int(protocol["seed"]))
    model.freeze()
    linear_to_lora_layers(model, int(protocol["num_layers"]), dict(protocol["lora"]))
    before = _trainable_copy(model)
    dataset = CompletionsDataset(
        [{"completion": str(example["completion"]), "prompt": str(example["prompt"])}],
        tokenizer,
        "prompt",
        "completion",
        True,
    )
    capture = Capture()
    adapter_dir = output / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    _write_mlx_adapter_config(adapter_dir, protocol)
    train(
        model=model,
        optimizer=optim.Adam(learning_rate=float(protocol["learning_rate"])),
        train_dataset=CacheDataset(dataset),
        val_dataset=None,
        args=TrainingArgs(
            batch_size=1,
            iters=1,
            val_batches=0,
            steps_per_report=1,
            steps_per_eval=10**9,
            steps_per_save=1,
            max_seq_length=int(protocol["max_seq_length"]),
            adapter_file=str(adapter_dir / "adapters.safetensors"),
            grad_accumulation_steps=1,
        ),
        training_callback=capture,
    )
    after = _trainable_copy(model)
    changed = before.keys() == after.keys() and not _same_leaves(before, after)
    reloaded, _tokenizer = load(local)
    reloaded.freeze()
    load_adapters(reloaded, str(adapter_dir))
    loaded = _trainable_copy(reloaded)
    saved = _trainable_copy(model)
    wall = time.perf_counter() - started
    peak = None if capture.peak_memory_gb is None else capture.peak_memory_gb * 1e9
    return {
        "backend": "mlx",
        "heldout_pack_used_for_update": False,
        "loss": capture.loss,
        "model_id": spec["repo_id"],
        "model_revision": spec["revision"],
        "optimizer_steps": 1,
        "reload_matches_saved": _same_leaves(saved, loaded),
        "resources": {"peak_memory_bytes": peak, "wall_seconds": wall},
        "train_seed": example["seed"],
        "trainable_parameters_changed": changed,
    }


def _load_causal(repo_id: str, revision: str, dtype: Any) -> Any:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    del torch
    try:
        return AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, torch_dtype=dtype)


def _masked_batch(tokenizer: Any, example: dict[str, object], device: Any) -> dict[str, Any]:
    prompt = str(example["prompt"])
    completion = str(example["completion"])
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("model tokenizer has no chat template")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    prompt_len = int(prompt_ids.shape[1])
    if prompt_len <= 0 or prompt_len >= int(input_ids.shape[1]):
        raise RuntimeError("prompt mask leaves no completion tokens")
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    batch = {key: value.to(device) for key, value in encoded.items()}
    batch["labels"] = labels.to(device)
    return batch


def _peak_bytes() -> float:
    import sys
    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated())
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(rss if sys.platform == "darwin" else rss * 1024)


def _transformers(protocol: dict[str, Any], example: dict[str, object], output: Path) -> dict[str, Any]:
    import torch  # type: ignore[import-not-found]
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    spec = protocol["models"]["transformers"]
    if int(protocol["num_layers"]) != 1 or int(protocol["optimizer_steps"]) != 1:
        raise RuntimeError("transformers smoke refuses to change the frozen step count")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec["repo_id"], revision=spec["revision"])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = _load_causal(spec["repo_id"], spec["revision"], dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    last_layer = model.config.num_hidden_layers - 1
    targets = [key.rsplit(".", 1)[-1] for key in protocol["lora"]["keys"]]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(protocol["lora"]["rank"]),
            lora_alpha=float(protocol["lora"]["scale"]),
            lora_dropout=float(protocol["lora"]["dropout"]),
            target_modules=targets,
            layers_to_transform=[last_layer],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    torch.manual_seed(int(protocol["seed"]))
    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    batch = _masked_batch(tokenizer, example, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(protocol["learning_rate"]))
    model.train()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    loss_value = float(loss.detach().cpu())
    changed = any(
        not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if name in before
    )
    adapter = output / "adapter"
    model.save_pretrained(adapter)
    saved = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    base = _load_causal(spec["repo_id"], spec["revision"], dtype)
    base.to(device)
    restored = PeftModel.from_pretrained(base, adapter)
    loaded = {
        name: parameter.detach().cpu().clone()
        for name, parameter in restored.named_parameters()
        if name in saved
    }
    matches = set(saved) == set(loaded) and all(torch.equal(saved[name], loaded[name]) for name in saved)
    return {
        "backend": "transformers",
        "heldout_pack_used_for_update": False,
        "loss": loss_value,
        "model_id": spec["repo_id"],
        "model_revision": spec["revision"],
        "optimizer_steps": 1,
        "reload_matches_saved": matches,
        "resources": {
            "peak_memory_bytes": _peak_bytes(),
            "wall_seconds": time.perf_counter() - started,
        },
        "train_seed": example["seed"],
        "trainable_parameters_changed": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="one-update exact-integer save/reload smoke")
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = _protocol()
    example = assert_train_example_is_outside_the_heldout_pack(PACK_PATH.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    environment: dict[str, object] = {
        "backend": args.backend,
        "python": platform.python_version(),
    }
    try:
        environment = _environment(args.backend)
        record = (
            _mlx(protocol, example, args.output)
            if args.backend == "mlx"
            else _transformers(protocol, example, args.output)
        )
    except Exception as exc:  # noqa: BLE001
        record = _failure(args.backend, exc)
    decision = accept_smoke_record(record)
    payload = {
        "code_revision": _code_revision(),
        "decision": decision,
        "environment": environment,
        "record": record,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (args.output / "smoke_result.json").write_text(text, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0 if decision["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
