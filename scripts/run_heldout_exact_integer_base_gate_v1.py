#!/usr/bin/env python3
"""Run the frozen held-out exact-integer pass@k gate.

Samples append to ``samples.jsonl`` and a finished ``summary.json`` is written
only when every task has k graded samples. Re-running the same output directory
skips completed samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.heldout_exact_integer import (  # noqa: E402
    MAX_NEW_TOKENS,
    TEMPERATURE,
    load_task_pack,
    read_jsonl,
    run_gate,
    verify_task_pack,
)

PROTOCOL_PATH = ROOT / "benchmarks/heldout/exact_integer_v1/protocol.json"


def _tasks(protocol: dict) -> list[dict[str, str]]:
    path = ROOT / protocol["tasks_file"]
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != protocol["tasks_sha256"]:
        raise SystemExit(f"task pack hash {digest} does not match the frozen protocol")
    rows = load_task_pack(blob.decode("utf-8"))
    verify_task_pack(rows)
    if len(rows) != int(protocol["n_tasks"]):
        raise SystemExit("task pack size does not match the frozen protocol")
    return rows


def _chat_prompt(tokenizer, prompt: str) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise RuntimeError("model tokenizer has no chat template; refusing an untemplated decode")
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _mlx_sampler(repo_id: str, revision: str):
    import mlx.core as mx  # type: ignore[import-not-found]
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    from mlx_lm import generate, load  # type: ignore[import-not-found]
    from mlx_lm.sample_utils import make_sampler  # type: ignore[import-not-found]

    local = snapshot_download(repo_id, revision=revision)
    model, tokenizer = load(local)
    sampler = make_sampler(temp=TEMPERATURE)

    def sample(prompt: str, seed: int) -> str:
        mx.random.seed(seed)
        text = generate(
            model,
            tokenizer,
            prompt=_chat_prompt(tokenizer, prompt),
            max_tokens=MAX_NEW_TOKENS,
            sampler=sampler,
            verbose=False,
        )
        print(f"sampled seed {seed}", flush=True)
        return text

    return sample


def _transformers_sampler(repo_id: str, revision: str):
    import torch  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=dtype)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, torch_dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def sample(prompt: str, seed: int) -> str:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        text = _chat_prompt(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        print(f"sampled seed {seed}", flush=True)
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return sample


def _record_environment(output: Path, backend: str) -> None:
    import platform

    record: dict[str, object] = {"backend": backend, "python": platform.python_version()}
    if backend == "mlx":
        import mlx_lm  # type: ignore[import-not-found]

        record["mlx_lm"] = getattr(mlx_lm, "__version__", "unknown")
    else:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]

        record["cuda"] = bool(torch.cuda.is_available())
        record["torch"] = torch.__version__
        record["transformers"] = transformers.__version__
    path = output / "environment.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise SystemExit("environment.json does not match this interpreter; refusing to append samples")
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def execute(protocol_path: Path, backend: str, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    spec = protocol["models"][backend]
    tasks = _tasks(protocol)
    sampler = (
        _mlx_sampler(spec["repo_id"], spec["revision"])
        if backend == "mlx"
        else _transformers_sampler(spec["repo_id"], spec["revision"])
    )
    _record_environment(output, backend)
    done = len(read_jsonl(output / "samples.jsonl"))
    print(f"completed samples on disk: {done} of {len(tasks) * 4}", flush=True)
    return run_gate(
        tasks,
        sampler,
        output,
        pack_sha256=protocol["tasks_sha256"],
        backend=backend,
        model_id=spec["repo_id"],
        protocol_id=str(protocol["protocol_id"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="held-out exact-integer base pass@k gate")
    parser.add_argument("--backend", choices=("mlx", "transformers"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = execute(PROTOCOL_PATH, args.backend, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
