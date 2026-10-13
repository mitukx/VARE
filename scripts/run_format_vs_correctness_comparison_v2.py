#!/usr/bin/env python3
"""Run the frozen format-versus-correctness comparison.

The confirm split is the decision set. The 64-task pack is a secondary curve.
Re-running the same output directory skips completed train steps and eval
samples. There is no flag that changes the step count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.format_vs_correctness_v2 import (  # noqa: E402
    GROUP_SIZE,
    MAX_NEW_TOKENS,
    MAX_SEQ_LENGTH,
    PROTOCOL_SHA256,
    RL_ARMS,
    RUN_SEEDS,
    SMOKE_MANIFEST_SHA256,
    TEMPERATURE,
    UPDATED_ARMS,
    assert_training_rows_are_disjoint,
    canonical_protocol_bytes,
    comparison_train_examples,
    confirm_eval_tasks,
    group_advantages,
    iter_eval_units,
    iter_train_units,
    pack_eval_tasks,
    resume_checkpoint,
    rl_rewards,
    rollout_seed,
    row_key,
    summarize_comparison,
    unit_key,
    validate_protocol,
)
from vare.heldout_exact_integer import append_jsonl, read_jsonl  # noqa: E402
from vare.format_vs_correctness_seeded_summary_v3 import (  # noqa: E402
    ANALYSIS_ID, ANALYSIS_SHA256, bind_analysis, read_progress_snapshot,
    summarize_seeded_comparison, validate_analysis,
)

PROTOCOL_PATH = ROOT / "protocols/format_vs_correctness_comparison_v2.json"
PACK_PATH = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
SPLIT_PATH = ROOT / "benchmarks/heldout/integer_seed_split_v1/splits.json"
SMOKE_MANIFEST = ROOT / "results/heldout-exact-integer-update-cost-smoke-v1/m1-mlx-f22b3b9/manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, Any]:
    blob = PROTOCOL_PATH.read_bytes()
    if blob != canonical_protocol_bytes() or _sha(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise SystemExit("comparison protocol bytes do not match the frozen hash")
    protocol = json.loads(blob.decode("utf-8"))
    validate_protocol(protocol)
    if _sha(SMOKE_MANIFEST) != SMOKE_MANIFEST_SHA256:
        raise SystemExit("retained smoke manifest does not match the comparison pin")
    if _sha(PACK_PATH) != protocol["heldout_pack_sha256"]:
        raise SystemExit("held-out pack bytes do not match the comparison pin")
    if _sha(SPLIT_PATH) != protocol["confirm_split_sha256"]:
        raise SystemExit("confirm split bytes do not match the comparison pin")
    assert_training_rows_are_disjoint(PACK_PATH.read_text(encoding="utf-8"))
    return protocol


def _tokens(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _require_template(tokenizer: Any) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("model tokenizer has no chat template")


def _encode_pair(tokenizer: Any, prompt: str, completion: str) -> tuple[list[int], int]:
    _require_template(tokenizer)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        add_generation_prompt=False,
        return_dict=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=False,
    )
    tokens = _tokens(full)
    offset = len(_tokens(prompt_ids))
    if offset < 1 or offset >= len(tokens) or len(tokens) > MAX_SEQ_LENGTH:
        raise RuntimeError("prompt mask leaves no completion tokens inside the frozen length")
    return tokens, offset


def _prompt_text(tokenizer: Any, prompt: str) -> str:
    _require_template(tokenizer)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _index(rows: list[dict[str, Any]], backend: str, model_id: str,
           *, analysis_sha: str | None = ANALYSIS_SHA256) -> dict[tuple[Any, ...], dict[str, Any]]:
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if analysis_sha is not None and row.get("analysis_sha256") != analysis_sha:
            raise SystemExit("progress has no matching prospective analysis pin; use retrospective analysis")
        if row.get("protocol_sha256") != PROTOCOL_SHA256:
            raise SystemExit("progress file was written for a different protocol hash")
        if row.get("backend") != backend or row.get("model_id") != model_id:
            raise SystemExit("progress file was written for a different backend or model")
        key = row_key(row)
        if key in found:
            raise SystemExit(f"duplicate progress row {key}")
        found[key] = row
    return found


def _completed_steps(rows: list[dict[str, Any]], arm: str, run_seed: int) -> list[int]:
    return [
        int(row["step"])
        for row in rows
        if row.get("kind") == "train" and row.get("arm") == arm and row.get("run_seed") == run_seed
    ]


def _write_mlx_adapter_config(directory: Path, protocol: dict[str, Any]) -> None:
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
    (directory / "adapter_config.json").write_text(text, encoding="utf-8")


def _mlx_nll(model: Any, tokens: list[int], offset: int) -> Any:
    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx.nn as nn  # type: ignore[import-not-found]

    batch = mx.array(tokens)[None]
    logits = model(batch)[:, :-1, :]
    targets = batch[:, 1:]
    steps = mx.arange(targets.shape[1])
    mask = steps >= (offset - 1)
    nll = nn.losses.cross_entropy(logits, targets) * mask
    return nll.sum() / mask.sum()


class _Mlx:
    def __init__(self, protocol: dict[str, Any], output: Path) -> None:
        import mlx.core as mx  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

        spec = protocol["models"]["mlx"]
        self.protocol = protocol
        self.output = output
        self.mx = mx
        self.local = snapshot_download(spec["repo_id"], revision=spec["revision"])
        self.model_id = spec["repo_id"]
        self.sampler = make_sampler(temp=TEMPERATURE)
        self.model: Any = None
        self.tokenizer: Any = None
        self.optimizer: Any = None
        self.position: tuple[Any, ...] | None = None

    def _checkpoint(self, arm: str, run_seed: int, step: int) -> Path:
        return self.output / "checkpoints" / arm / str(run_seed) / f"step-{step:04d}"

    def _load_base(self) -> None:
        from mlx_lm.utils import load  # type: ignore[import-not-found]

        self.model, self.tokenizer = load(self.local)
        self.model.freeze()
        self.optimizer = None

    def prepare_train(self, arm: str, run_seed: int, step: int) -> None:
        if self.position == ("train", arm, run_seed, step - 1):
            return
        import mlx.optimizers as optim  # type: ignore[import-not-found]
        from mlx.utils import tree_unflatten  # type: ignore[import-not-found]
        from mlx_lm.tuner.utils import linear_to_lora_layers, load_adapters  # type: ignore[import-not-found]

        self._load_base()
        learning_rate = float(self.protocol["learning_rate"])
        if step == 1:
            self.mx.random.seed(run_seed)
            linear_to_lora_layers(self.model, int(self.protocol["num_layers"]), dict(self.protocol["lora"]))
            self.optimizer = optim.Adam(learning_rate=learning_rate)
        else:
            directory = self._checkpoint(arm, run_seed, step - 1)
            load_adapters(self.model, str(directory))
            self.optimizer = optim.Adam(learning_rate=learning_rate)
            state = self.mx.load(str(directory / "optimizer.safetensors"))
            self.optimizer.state = tree_unflatten(list(state.items()))
        self.position = ("train", arm, run_seed, step - 1)

    def _save(self, arm: str, run_seed: int, step: int) -> None:
        from mlx.utils import tree_flatten  # type: ignore[import-not-found]

        directory = self._checkpoint(arm, run_seed, step)
        directory.mkdir(parents=True, exist_ok=True)
        _write_mlx_adapter_config(directory, self.protocol)
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        self.mx.eval(*weights.values())
        self.mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        flat = dict(tree_flatten(self.optimizer.state))
        self.mx.eval(*flat.values())
        self.mx.save_safetensors(str(directory / "optimizer.safetensors"), flat)

    def _step(self, loss_fn: Any) -> float:
        import mlx.nn as nn  # type: ignore[import-not-found]

        self.model.train()
        value_and_grad = nn.value_and_grad(self.model, loss_fn)
        loss, grad = value_and_grad(self.model)
        self.optimizer.update(self.model, grad)
        self.mx.eval(self.model.parameters(), self.optimizer.state, loss)
        return float(loss.item())

    def train(self, unit: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        arm = str(unit["arm"])
        run_seed = int(unit["run_seed"])
        step = int(unit["step"])
        self.prepare_train(arm, run_seed, step)
        started = time.perf_counter()
        prompt = str(example["prompt"])
        outputs: list[str] | None = None
        rewards: list[float] | None = None
        advantages: list[float] | None = None
        if arm == "format_sft":
            tokens, offset = _encode_pair(self.tokenizer, prompt, str(example["completion"]))

            def sft_loss(model: Any, tokens: list[int] = tokens, offset: int = offset) -> Any:
                return _mlx_nll(model, tokens, offset)

            loss = self._step(sft_loss)
        elif arm in RL_ARMS:
            outputs = [self.generate(prompt, rollout_seed(arm, run_seed, step, index)) for index in range(GROUP_SIZE)]
            rewards = rl_rewards(arm, str(example["expression"]), outputs, run_seed, step)
            advantages = group_advantages(rewards)
            encoded_rows = [_encode_pair(self.tokenizer, prompt, text) for text in outputs]
            advantage_rows = advantages

            def rl_loss(model: Any) -> Any:
                total = self.mx.array(0.0)
                for (tokens, offset), advantage in zip(encoded_rows, advantage_rows, strict=True):
                    total = total + self.mx.array(advantage) * _mlx_nll(model, tokens, offset)
                return total / len(encoded_rows)

            self.prepare_train(arm, run_seed, step)
            loss = self._step(rl_loss)
        else:
            raise RuntimeError(f"unknown training arm {arm}")
        self._save(arm, run_seed, step)
        self.position = ("train", arm, run_seed, step)
        return _train_row(arm, run_seed, step, example, loss, outputs, rewards, advantages, started, "mlx", self.model_id)

    def prepare_eval(self, arm: str, run_seed: int, checkpoint_step: int) -> None:
        key = ("eval", arm, run_seed, checkpoint_step)
        if self.position == key:
            return
        from mlx_lm.tuner.utils import load_adapters  # type: ignore[import-not-found]

        self._load_base()
        if arm != "base":
            directory = self._checkpoint(arm, run_seed, checkpoint_step)
            load_adapters(self.model, str(directory))
        self.model.eval()
        self.position = key

    def generate(self, prompt: str, seed: int) -> str:
        from mlx_lm import generate  # type: ignore[import-not-found]

        self.model.eval()
        self.mx.random.seed(seed)
        return generate(
            self.model,
            self.tokenizer,
            prompt=_prompt_text(self.tokenizer, prompt),
            max_tokens=MAX_NEW_TOKENS,
            sampler=self.sampler,
            verbose=False,
        )


def _load_causal(repo_id: str, revision: str, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    try:
        return AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, torch_dtype=dtype)


def _tf_nll(model: Any, tokens: list[int], offset: int, device: Any) -> Any:
    import torch  # type: ignore[import-not-found]

    ids = torch.tensor([tokens], device=device)
    logits = model(input_ids=ids).logits[:, :-1, :]
    gathered = torch.log_softmax(logits, dim=-1).gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    mask = torch.arange(gathered.shape[1], device=device) >= (offset - 1)
    if int(mask.sum().item()) == 0:
        raise RuntimeError("prompt mask leaves no completion tokens")
    return -(gathered * mask).sum() / mask.sum()


class _Transformers:
    def __init__(self, protocol: dict[str, Any], output: Path) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        spec = protocol["models"]["transformers"]
        self.protocol = protocol
        self.output = output
        self.torch = torch
        self.model_id = spec["repo_id"]
        self.revision = spec["revision"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        _require_template(self.tokenizer)
        self.model: Any = None
        self.optimizer: Any = None
        self.position: tuple[Any, ...] | None = None

    def _checkpoint(self, arm: str, run_seed: int, step: int) -> Path:
        return self.output / "checkpoints" / arm / str(run_seed) / f"step-{step:04d}"

    def _lora(self, base: Any) -> Any:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

        last_layer = base.config.num_hidden_layers - 1
        targets = [key.rsplit(".", 1)[-1] for key in self.protocol["lora"]["keys"]]
        return get_peft_model(
            base,
            LoraConfig(
                r=int(self.protocol["lora"]["rank"]),
                lora_alpha=float(self.protocol["lora"]["scale"]),
                lora_dropout=float(self.protocol["lora"]["dropout"]),
                target_modules=targets,
                layers_to_transform=[last_layer],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

    def prepare_train(self, arm: str, run_seed: int, step: int) -> None:
        if self.position == ("train", arm, run_seed, step - 1):
            return
        from peft import PeftModel  # type: ignore[import-not-found]

        base = _load_causal(self.model_id, self.revision, self.dtype).to(self.device)
        if step == 1:
            self.torch.manual_seed(run_seed)
            self.model = self._lora(base)
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            self.optimizer = self.torch.optim.Adam(trainable, lr=float(self.protocol["learning_rate"]))
        else:
            directory = self._checkpoint(arm, run_seed, step - 1)
            self.model = PeftModel.from_pretrained(base, directory).to(self.device)
            trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            self.optimizer = self.torch.optim.Adam(trainable, lr=float(self.protocol["learning_rate"]))
            state = self.torch.load(directory / "optimizer.pt", map_location=self.device)
            self.optimizer.load_state_dict(state)
        self.position = ("train", arm, run_seed, step - 1)

    def _save(self, arm: str, run_seed: int, step: int) -> None:
        directory = self._checkpoint(arm, run_seed, step)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")

    def _step(self, loss: Any) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def train(self, unit: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
        arm = str(unit["arm"])
        run_seed = int(unit["run_seed"])
        step = int(unit["step"])
        self.prepare_train(arm, run_seed, step)
        started = time.perf_counter()
        prompt = str(example["prompt"])
        outputs: list[str] | None = None
        rewards: list[float] | None = None
        advantages: list[float] | None = None
        if arm == "format_sft":
            tokens, offset = _encode_pair(self.tokenizer, prompt, str(example["completion"]))
            loss = self._step(_tf_nll(self.model, tokens, offset, self.device))
        elif arm in RL_ARMS:
            outputs = [self.generate(prompt, rollout_seed(arm, run_seed, step, index)) for index in range(GROUP_SIZE)]
            rewards = rl_rewards(arm, str(example["expression"]), outputs, run_seed, step)
            advantages = group_advantages(rewards)
            encoded = [_encode_pair(self.tokenizer, prompt, text) for text in outputs]
            total = self.torch.zeros((), device=self.device)
            for (tokens, offset), advantage in zip(encoded, advantages, strict=True):
                weight = self.torch.tensor(advantage, device=self.device)
                total = total + weight * _tf_nll(self.model, tokens, offset, self.device)
            self.prepare_train(arm, run_seed, step)
            loss = self._step(total / len(encoded))
        else:
            raise RuntimeError(f"unknown training arm {arm}")
        self._save(arm, run_seed, step)
        self.position = ("train", arm, run_seed, step)
        return _train_row(
            arm, run_seed, step, example, loss, outputs, rewards, advantages, started, "transformers", self.model_id
        )

    def prepare_eval(self, arm: str, run_seed: int, checkpoint_step: int) -> None:
        key = ("eval", arm, run_seed, checkpoint_step)
        if self.position == key:
            return
        from peft import PeftModel  # type: ignore[import-not-found]

        base = _load_causal(self.model_id, self.revision, self.dtype).to(self.device)
        if arm == "base":
            self.model = base
        else:
            directory = self._checkpoint(arm, run_seed, checkpoint_step)
            self.model = PeftModel.from_pretrained(base, directory).to(self.device)
        self.model.eval()
        self.position = key

    def generate(self, prompt: str, seed: int) -> str:
        self.model.eval()
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        text = _prompt_text(self.tokenizer, prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _train_row(
    arm: str,
    run_seed: int,
    step: int,
    example: dict[str, Any],
    loss: float,
    outputs: list[str] | None,
    rewards: list[float] | None,
    advantages: list[float] | None,
    started: float,
    backend: str,
    model_id: str,
) -> dict[str, Any]:
    return {
        "analysis_sha256": ANALYSIS_SHA256,
        "advantages": advantages,
        "arm": arm,
        "backend": backend,
        "kind": "train",
        "loss": loss,
        "model_id": model_id,
        "outputs": outputs,
        "protocol_sha256": PROTOCOL_SHA256,
        "rewards": rewards,
        "run_seed": run_seed,
        "step": step,
        "train_seed": example["seed"],
        "wall_seconds": time.perf_counter() - started,
    }


def _eval_row(
    unit: dict[str, Any],
    output: str,
    started: float,
    backend: str,
    model_id: str,
) -> dict[str, Any]:
    return {
        "analysis_sha256": ANALYSIS_SHA256,
        "arm": unit["arm"],
        "backend": backend,
        "checkpoint_step": unit["checkpoint_step"],
        "kind": "eval",
        "model_id": model_id,
        "output": output,
        "protocol_sha256": PROTOCOL_SHA256,
        "run_seed": unit["run_seed"],
        "sample_index": unit["sample_index"],
        "sample_seed": unit["sample_seed"],
        "split": unit["split"],
        "task_id": unit["task_id"],
        "wall_seconds": time.perf_counter() - started,
    }


def _environment(backend: str) -> dict[str, object]:
    record: dict[str, object] = {"backend": backend, "python": platform.python_version()}
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


def execute(backend: str, output: Path) -> int:
    protocol = _protocol()
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks(PACK_PATH.read_text(encoding="utf-8"))
    examples = {int(row["seed"]): row for row in comparison_train_examples()}
    model_id = str(protocol["models"][backend]["repo_id"])
    output.mkdir(parents=True, exist_ok=True)
    bind_analysis(output)
    environment_path = output / "environment.json"
    environment = json.dumps(_environment(backend), indent=2, sort_keys=True) + "\n"
    if environment_path.exists() and environment_path.read_text(encoding="utf-8") != environment:
        raise SystemExit("environment.json does not match this interpreter")
    environment_path.write_text(environment, encoding="utf-8")
    progress = output / "progress.jsonl"
    rows = read_jsonl(progress)
    for arm in UPDATED_ARMS:
        for run_seed in RUN_SEEDS:
            resume_checkpoint(_completed_steps(rows, arm, run_seed))
    done = _index(rows, backend, model_id)
    session: Any = _Mlx(protocol, output) if backend == "mlx" else _Transformers(protocol, output)
    for unit in iter_train_units():
        key = unit_key(unit)
        if key in done:
            continue
        example = examples[int(unit["train_seed"])]
        row = session.train(unit, example)
        append_jsonl(progress, row)
        rows.append(row)
        done[key] = row
        print(f"train {unit['arm']} seed {unit['run_seed']} step {unit['step']}", flush=True)
    tasks = {row["task_id"]: row for row in [*confirm, *pack]}
    for unit in iter_eval_units(confirm, pack):
        key = unit_key(unit)
        if key in done:
            continue
        session.prepare_eval(str(unit["arm"]), int(unit["run_seed"]), int(unit["checkpoint_step"]))
        started = time.perf_counter()
        text = session.generate(tasks[str(unit["task_id"])]["prompt"], int(unit["sample_seed"]))
        row = _eval_row(unit, text, started, backend, model_id)
        append_jsonl(progress, row)
        rows.append(row)
        done[key] = row
        print(
            f"eval {unit['arm']} seed {unit['run_seed']} step {unit['checkpoint_step']} {unit['task_id']} #{unit['sample_index']}",
            flush=True,
        )
    # Bind both reports to the same captured progress bytes and digest.
    rows, progress_sha = read_progress_snapshot(progress)
    _index(rows, backend, model_id)
    summary = summarize_comparison(confirm, pack, rows)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (output / "summary.json").write_text(text, encoding="utf-8")
    seeded = summarize_seeded_comparison(confirm, pack, rows, prospective=True)
    seeded["analysis_file_pin_verified"] = True
    seeded["progress_sha256"] = progress_sha
    (output / "summary.seeded-v3.json").write_text(
        json.dumps(seeded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "protocol_id": summary["protocol_id"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="frozen format versus correctness comparison")
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    validate_analysis()
    protocol = _protocol()
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks(PACK_PATH.read_text(encoding="utf-8"))
    train_units = iter_train_units()
    eval_units = iter_eval_units(confirm, pack)
    if args.check:
        print(
            json.dumps(
                {
                    "analysis_id": ANALYSIS_ID,
                    "analysis_sha256": ANALYSIS_SHA256,
                    "eval_units": len(eval_units),
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": PROTOCOL_SHA256,
                    "status": "checked",
                    "train_units": len(train_units),
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        return execute(args.backend, args.output)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
