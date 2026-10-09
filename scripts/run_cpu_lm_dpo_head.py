#!/usr/bin/env python3
"""Bounded offline CPU DPO-style update on an already-cached causal LM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_dpo_head_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_dpo_head_v1.lock.json"
TEMPLATES = json.loads(SPEC_PATH.read_text(encoding="utf-8"))["task"]["prompt_templates"]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_locked_protocol() -> tuple[dict[str, Any], str]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_hash = lock.pop("sha256", None)
    observed_hash = sha256_bytes(canonical(spec))
    if locked_hash != observed_hash or canonical(lock) != canonical(spec):
        raise ValueError("protocol differs from locked specification")
    return spec, observed_hash


def generate_examples(seed: int, split: str, count: int) -> list[dict[str, Any]]:
    is_train = split == "train"
    rng = random.Random((10000 if is_train else 20000) + seed)
    low, high = (1, 19) if is_train else (20, 39)
    pairs: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    while len(rows) < count:
        left, right = rng.randint(low, high), rng.randint(low, high)
        pair = tuple(sorted((left, right)))
        if left == right or pair in pairs:
            continue
        pairs.add(pair)
        total = left + right
        wrong = total - 1 if total > 1 else total + 1
        correct_label = rng.choice(["A", "B"])
        candidate_a, candidate_b = (total, wrong) if correct_label == "A" else (wrong, total)
        template_index = rng.randrange(len(TEMPLATES))
        prompt = TEMPLATES[template_index].format(
            left=left, right=right, candidate_a=candidate_a, candidate_b=candidate_b
        )
        rows.append({
            "split": split,
            "seed": seed,
            "left": left,
            "right": right,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "correct_label": correct_label,
            "template_index": template_index,
            "prompt": prompt,
        })
    return rows


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output: Path) -> None:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(output).as_posix()] = sha256_file(path)
    json_write(output / "manifest.json", {"algorithm": "sha256", "files": files})


def bootstrap_paired(values: list[float], seed: int, resamples: int) -> list[float]:
    """Seed-stratified bootstrap of per-example updated-minus-base NLL."""
    grouped: dict[int, list[float]] = {}
    for seed_value, delta in values:
        grouped.setdefault(seed_value, []).append(delta)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        per_seed = [statistics.fmean(rng.choices(group, k=len(group))) for group in grouped.values()]
        means.append(statistics.fmean(per_seed))
    means.sort()
    return [means[math.floor(.025 * (resamples - 1))], means[math.floor(.975 * (resamples - 1))]]


def preference_metrics(margins: list[float], signs: list[int]) -> dict[str, float]:
    nll = [max(-sign * margin, 0.0) + math.log1p(math.exp(-abs(sign * margin))) for margin, sign in zip(margins, signs)]
    correct = [1.0 if sign * margin > 0 else 0.5 if margin == 0 else 0.0 for margin, sign in zip(margins, signs)]
    return {"mean_conditional_preference_nll": statistics.fmean(nll),
            "preference_accuracy": statistics.fmean(correct)}


def kl_to_base(base_margins: list[float], updated_margins: list[float]) -> float:
    total = 0.0
    for base, updated in zip(base_margins, updated_margins):
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, base))))
        q = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, updated))))
        total += q * math.log(max(q, 1e-300) / max(p, 1e-300))
        total += (1.0 - q) * math.log(max(1.0 - q, 1e-300) / max(1.0 - p, 1e-300))
    return total / len(base_margins)


class ResourceGuard:
    def __init__(self, max_seconds: int, max_rss_bytes: int):
        self.max_seconds = max_seconds
        self.max_rss_bytes = max_rss_bytes
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.previous_usr1 = None
        self.previous_alarm = None

    def __enter__(self):
        def abort(signum, frame):
            raise RuntimeError(f"resource guard aborted run (signal {signum})")

        self.previous_usr1 = signal.signal(signal.SIGUSR1, abort)
        self.previous_alarm = signal.signal(signal.SIGALRM, abort)
        signal.alarm(self.max_seconds)

        def monitor():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.max_rss_bytes:
                    os.kill(os.getpid(), signal.SIGUSR1)

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
        signal.alarm(0)
        signal.signal(signal.SIGUSR1, self.previous_usr1)
        signal.signal(signal.SIGALRM, self.previous_alarm)


def run(spec: dict[str, Any], spec_hash: str, model_dir: Path, output: Path) -> dict[str, Any]:
    run_started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    for name, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HF_HUB_DISABLE_TELEMETRY", "1")):
        os.environ[name] = value
    try:
        with ResourceGuard(spec["compute_limits"]["max_wall_seconds"], spec["compute_limits"]["max_peak_rss_bytes"]):
            import torch
            import numpy
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            observed_runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                                "transformers": transformers.__version__, "numpy": numpy.__version__}
            if observed_runtime != spec["runtime"]:
                raise RuntimeError(f"runtime differs from frozen versions: {observed_runtime}")
            torch.set_num_threads(spec["compute_limits"]["threads"])
            torch.manual_seed(0)
            if torch.cuda.is_initialized():
                raise RuntimeError("CUDA must not be initialized")
            revision = model_dir.name
            if revision != spec["model"]["revision"]:
                raise ValueError(f"model path revision {revision!r} does not match locked revision")
            required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
            missing = [name for name in required if not (model_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"cached model files missing: {missing}")
            model_file_hashes = {name: sha256_file(model_dir / name) for name in required}
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            tokenizer.padding_side = "right"
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("tokenizer has neither a padding nor EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), local_files_only=True, torch_dtype=torch.float32
            )
            model.to(torch.device("cpu"))
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
                if parameter.device.type != "cpu":
                    raise RuntimeError("non-CPU model parameter detected")
            if not hasattr(model, "model") or not hasattr(model, "lm_head"):
                raise TypeError("expected a causal LM exposing .model and .lm_head")

            token_a = tokenizer.encode(" A", add_special_tokens=False)
            token_b = tokenizer.encode(" B", add_special_tokens=False)
            if len(token_a) != 1 or len(token_b) != 1 or token_a[0] == token_b[0]:
                raise ValueError("the locked response labels must each map to one distinct token")
            token_ids = [token_a[0], token_b[0]]
            results: list[dict[str, Any]] = []
            paired_nll_deltas: list[tuple[int, float]] = []
            data_record: dict[str, Any] = {}
            source_digest = sha256_file(Path(__file__))
            def encode_hidden(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
                batch = tokenizer([row["prompt"] for row in rows], return_tensors="pt", padding=True,
                                  add_special_tokens=True, truncation=False)
                input_ids = batch["input_ids"].to("cpu")
                attention = batch["attention_mask"].to("cpu")
                lengths = attention.sum(dim=1) - 1
                # Keep frozen activations as ordinary no-grad tensors. PyTorch inference tensors
                # cannot be saved by autograd when the trainable head adapter consumes them.
                with torch.no_grad():
                    hidden_all = model.model(input_ids=input_ids, attention_mask=attention,
                                             use_cache=False).last_hidden_state
                    indices = torch.arange(hidden_all.shape[0])
                    hidden = hidden_all[indices, lengths].to(dtype=torch.float32).contiguous()
                    weight_rows = model.lm_head.weight[token_ids].to(dtype=torch.float32)
                    base_logits = torch.nn.functional.linear(hidden, weight_rows)
                return hidden, base_logits

            for seed in spec["learner"]["seeds"]:
                seed_start = time.monotonic()
                train_rows = generate_examples(seed, "train", spec["task"]["train"]["examples_per_seed"])
                test_rows = generate_examples(seed, "heldout", spec["task"]["heldout"]["examples_per_seed"])
                train_h, train_base = encode_hidden(train_rows)
                test_h, test_base = encode_hidden(test_rows)
                train_sign = torch.tensor([1.0 if row["correct_label"] == "A" else -1.0 for row in train_rows])
                test_signs = [1 if row["correct_label"] == "A" else -1 for row in test_rows]
                generator = torch.Generator(device="cpu").manual_seed(seed)
                hidden_size = train_h.shape[1]
                rank = 4
                adapter_a = torch.nn.Parameter(torch.randn((hidden_size, rank), generator=generator) / math.sqrt(hidden_size))
                adapter_b = torch.nn.Parameter(torch.zeros((rank, 2)))
                adapter_a_initial, adapter_b_initial = adapter_a.detach().clone(), adapter_b.detach().clone()
                optimizer = torch.optim.SGD([adapter_a, adapter_b], lr=spec["learner"]["optimizer"]["learning_rate"],
                                            weight_decay=spec["learner"]["optimizer"]["weight_decay"])
                beta = spec["learner"]["beta"]
                updates = spec["learner"]["updates"]
                train_losses = []
                for _ in range(updates):
                    optimizer.zero_grad(set_to_none=True)
                    delta = (train_h @ adapter_a) @ adapter_b
                    delta_margin = delta[:, 0] - delta[:, 1]
                    loss = torch.nn.functional.softplus(-beta * train_sign * delta_margin).mean()
                    loss.backward()
                    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in (adapter_a, adapter_b)):
                        raise FloatingPointError("non-finite DPO loss or adapter gradient")
                    train_losses.append(float(loss.detach()))
                    optimizer.step()

                with torch.inference_mode():
                    train_delta = ((train_h @ adapter_a) @ adapter_b)
                    test_delta = ((test_h @ adapter_a) @ adapter_b)
                    train_base_margin = (train_base[:, 0] - train_base[:, 1]).tolist()
                    test_base_margin = (test_base[:, 0] - test_base[:, 1]).tolist()
                    train_updated_margin = (train_base[:, 0] - train_base[:, 1] + train_delta[:, 0] - train_delta[:, 1]).tolist()
                    test_updated_margin = (test_base[:, 0] - test_base[:, 1] + test_delta[:, 0] - test_delta[:, 1]).tolist()
                    adapter_a_final, adapter_b_final = adapter_a.detach().clone(), adapter_b.detach().clone()
                base_metrics = preference_metrics(test_base_margin, test_signs)
                updated_metrics = preference_metrics(test_updated_margin, test_signs)
                for before_margin, after_margin, sign in zip(test_base_margin, test_updated_margin, test_signs):
                    before_nll = max(-sign * before_margin, 0.0) + math.log1p(math.exp(-abs(sign * before_margin)))
                    after_nll = max(-sign * after_margin, 0.0) + math.log1p(math.exp(-abs(sign * after_margin)))
                    paired_nll_deltas.append((seed, after_nll - before_nll))
                train_base_margin_tensor = train_base[:, 0] - train_base[:, 1]
                train_updated_margin_tensor = torch.tensor(train_updated_margin)
                train_base_margin_list = train_base_margin_tensor.tolist()
                train_base_metrics = preference_metrics(train_base_margin_list, [int(v) for v in train_sign.tolist()])
                train_updated_metrics = preference_metrics(train_updated_margin, [int(v) for v in train_sign.tolist()])
                train_objective_after = float(torch.nn.functional.softplus(-beta * train_sign * (train_updated_margin_tensor - train_base_margin_tensor)).mean())
                seed_record = {
                    "seed": seed,
                    "train_examples": train_rows,
                    "heldout_examples": test_rows,
                    "train_base_margins": train_base_margin,
                    "train_updated_margins": train_updated_margin,
                    "heldout_base_margins": test_base_margin,
                    "heldout_updated_margins": test_updated_margin,
                    "metrics": {
                        "heldout_base": base_metrics,
                        "heldout_updated": updated_metrics,
                        "heldout_mean_bernoulli_kl_updated_to_base": kl_to_base(test_base_margin, test_updated_margin),
                        "train_objective_first_update": train_losses[0],
                        "train_objective_last_update": train_losses[-1],
                        "train_objective_after": train_objective_after,
                        "train_base_conditional_preference_nll": train_base_metrics["mean_conditional_preference_nll"],
                        "train_updated_conditional_preference_nll": train_updated_metrics["mean_conditional_preference_nll"],
                        "adapter_l2": float(torch.sqrt(adapter_a_final.square().sum() + adapter_b_final.square().sum())),
                        "adapter_update_l2": float(torch.sqrt((adapter_a_final - adapter_a_initial).square().sum() + (adapter_b_final - adapter_b_initial).square().sum())),
                        "updates": updates,
                        "wall_seconds": time.monotonic() - seed_start,
                        "peak_rss_bytes": rss_bytes(),
                        "device": "cpu"
                    },
                    "adapter": {"A_initial": adapter_a_initial.tolist(), "B_initial": adapter_b_initial.tolist(),
                                "A_final": adapter_a_final.tolist(), "B_final": adapter_b_final.tolist()}
                }
                data_record[str(seed)] = seed_record
                results.append(seed_record["metrics"] | {"seed": seed})
                if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                    raise MemoryError("peak RSS exceeded frozen limit")

            mean_change = statistics.fmean(value for _, value in paired_nll_deltas)
            interval = bootstrap_paired(paired_nll_deltas, spec["metrics"]["paired_bootstrap"]["seed"],
                                       spec["metrics"]["paired_bootstrap"]["resamples"])
            all_improve = all(row["heldout_updated"]["mean_conditional_preference_nll"] < row["heldout_base"]["mean_conditional_preference_nll"] for row in results)
            all_kl_ok = all(row["heldout_mean_bernoulli_kl_updated_to_base"] <= 0.5 for row in results)
            passed = all_improve and interval[1] < 0 and all_kl_ok
            summary = {
                "protocol_id": spec["protocol_id"], "protocol_sha256": spec_hash,
                "runner_sha256": source_digest, "model_id": spec["model"]["id"],
                "model_revision": revision, "model_files_sha256": model_file_hashes,
                "response_token_ids": {"A": token_ids[0], "B": token_ids[1]},
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": numpy.__version__,
                "device": "cpu", "paid_compute": False, "network_disabled": True,
                "per_seed": results, "mean_heldout_nll_change_updated_minus_base": mean_change,
                "paired_seed_stratified_bootstrap_95pct": interval,
                "all_seeds_improve": all_improve, "all_seed_kl_within_limit": all_kl_ok,
                "decision": "positive_small_model_preference_result" if passed else "non_pass_or_null",
                "total_wall_seconds": time.monotonic() - run_started, "peak_rss_bytes": rss_bytes(),
                "compute_limits_respected": rss_bytes() <= spec["compute_limits"]["max_peak_rss_bytes"]
            }
            json_write(output / "seed_records.json", data_record)
            json_write(output / "summary.json", summary)
            json_write(output / "environment.json", {"python_executable": sys.executable,
                "environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")},
                "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                    capture_output=True, check=False).stdout.strip()})
            return summary
    except BaseException as exc:
        json_write(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.monotonic() - run_started,
                                              "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the already-cached immutable snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, protocol_hash = read_locked_protocol()
    summary = run(spec, protocol_hash, args.model_dir.expanduser().resolve(), args.output.expanduser().resolve())
    write_manifest(args.output.resolve())
    print(json.dumps({"decision": summary["decision"], "mean_nll_change": summary["mean_heldout_nll_change_updated_minus_base"],
                      "interval": summary["paired_seed_stratified_bootstrap_95pct"], "wall_seconds": summary["total_wall_seconds"],
                      "peak_rss_bytes": summary["peak_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
