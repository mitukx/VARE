#!/usr/bin/env python3
"""Run a bounded Qwen3-4B MLX GRPO experiment on parent-scored function tasks.

This runner sends only task instructions to the model. Candidate calls use the
existing isolated Docker worker; expected values and strict JSON verdict logic
remain in the parent process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vare.container_isolation import DockerIsolationRunner  # noqa: E402
from vare.evidence import sha256_json  # noqa: E402
from vare import grpo_mlx  # noqa: E402
from vare import function_repair_tasks as task_pack  # noqa: E402
from vare.isolated_candidate import _one_call  # noqa: E402

MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
BASE_MODEL_ID = f"{MODEL_ID}@{MODEL_REVISION}"
CONTAINER_IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
SYSTEM_PROMPT = (
    "You are solving a bounded Python function-repair task. Return only Python source defining "
    "`def solve(...)`. Use only Python built-ins and do not include imports, Markdown fences, "
    "explanations, or tests. Follow the task contract exactly."
)
MAX_PROMPT_TOKENS = 2048
TRAIN_MAX_NEW_TOKENS = 512
DEVELOPMENT_MAX_NEW_TOKENS = 256
GROUP_SIZE = 4
TRAIN_SAMPLES_PER_TASK = 4
RECOVERY_TASK_IDS = (
    "train.maximum_fixed_window.v1",
    "train.parse_env_lines.v1",
    "train.valid_brackets.v1",
)
GRPO_CONFIG = grpo_mlx.GRPOConfig(
    clip_eps_low=0.2,
    clip_eps_high=0.2,
    normalize_std=True,
    temperature=0.7,
    max_policy_lag=0,
    max_grad_norm=1.0,
    group_size=GROUP_SIZE,
)
LORA_CONFIG = {"num_layers": 8, "rank": 8, "scale": 20.0, "dropout": 0.0}
OPTIMIZER_CONFIG = {"name": "mlx.optimizers.Adam", "learning_rate": 1e-5}
DISK_MIN_GIB = 8.0
AVAILABLE_MEMORY_MIN_GIB = 8.0
MEMORY_FREE_MIN_PCT = 15.0
START_SWAP_MAX = 0.94
STOP_SWAP_MAX = 0.97
MAX_RUN_SECONDS = 7200.0
EVAL_TIMEOUT_SECONDS = 5.0
EVAL_MEMORY_MB = 512
GIB = 1024**3
DEFAULT_PROTOCOL = ROOT / "protocols/qwen3_function_repair_grpo_dev_v1.json"


class ResourceStop(RuntimeError):
    """A frozen resource limit fired before an optimizer mutation."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_directory(path: Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        h.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(bytes.fromhex(sha256_file(item)))
    return h.hexdigest()


def digest_tree(tree: Any) -> str:
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    mx.eval(tree)
    h = hashlib.sha256()
    for key, value in sorted(tree_flatten(tree)):
        h.update(str(key).encode("utf-8") + b"\0")
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def optimizer_digest(optimizer: Any) -> str:
    return digest_tree(optimizer.state)


def process_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def _vm_available_memory_bytes(output: str) -> int | None:
    page_match = re.search(r"page size of (\d+) bytes", output)
    if page_match is None:
        return None
    values = {}
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{re.escape(label)}:\s+(\d+)", output, flags=re.MULTILINE)
        if match is None:
            return None
        values[label] = int(match.group(1))
    return int(page_match.group(1)) * sum(values.values())


def available_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        if platform.system() != "Darwin":
            return None
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False)
        return _vm_available_memory_bytes(result.stdout) if result.returncode == 0 else None


def _memory_free_pct() -> float | None:
    result = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, check=False)
    match = re.search(r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%", result.stdout)
    return float(match.group(1)) if match else None


def _swap_usage() -> dict[str, float] | None:
    if platform.system() != "Darwin":
        return None
    result = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False)
    match = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", result.stdout)
    if match is None:
        return None
    total, used = (float(value) for value in match.groups())
    return {"total_gib": total / 1024, "used_gib": used / 1024,
            "fraction": used / total if total else 0.0}


def resource_snapshot(path: Path) -> dict[str, Any]:
    available = available_memory_bytes()
    disk = shutil.disk_usage(path)
    swap = _swap_usage()
    return {
        "time": time.time(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "available_memory_gib": available / GIB if available is not None else None,
        "memory_free_pct": _memory_free_pct(),
        "disk_free_gib": disk.free / GIB,
        "swap": swap,
        "process_cpu_seconds": process_cpu_seconds(),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB,
    }


def preflight(path: Path, phase: str) -> dict[str, Any]:
    snap = resource_snapshot(path)
    if snap["disk_free_gib"] < DISK_MIN_GIB:
        raise ResourceStop(f"free disk {snap['disk_free_gib']:.2f} GiB < {DISK_MIN_GIB:.2f} GiB")
    if snap["available_memory_gib"] is not None and snap["available_memory_gib"] < AVAILABLE_MEMORY_MIN_GIB:
        raise ResourceStop(
            f"available memory {snap['available_memory_gib']:.2f} GiB < {AVAILABLE_MEMORY_MIN_GIB:.2f} GiB"
        )
    if snap["memory_free_pct"] is not None and snap["memory_free_pct"] < MEMORY_FREE_MIN_PCT:
        raise ResourceStop(f"free memory pressure {snap['memory_free_pct']:.0f}% < {MEMORY_FREE_MIN_PCT:.0f}%")
    if snap["swap"] is not None and snap["swap"]["fraction"] >= START_SWAP_MAX:
        raise ResourceStop(f"swap fraction {snap['swap']['fraction']:.4f} >= {START_SWAP_MAX:.4f}")
    atomic_json(path / f"preflight-{phase}.json", snap)
    return snap


def resource_guard(path: Path, phase_started: float) -> dict[str, Any]:
    if time.monotonic() - phase_started > MAX_RUN_SECONDS:
        raise ResourceStop(f"phase exceeded {MAX_RUN_SECONDS:.0f} second wall limit")
    snap = resource_snapshot(path)
    if snap["disk_free_gib"] < DISK_MIN_GIB:
        raise ResourceStop(f"free disk fell below {DISK_MIN_GIB:.2f} GiB")
    if snap["available_memory_gib"] is not None and snap["available_memory_gib"] < AVAILABLE_MEMORY_MIN_GIB:
        raise ResourceStop(f"available memory fell below {AVAILABLE_MEMORY_MIN_GIB:.2f} GiB")
    if snap["memory_free_pct"] is not None and snap["memory_free_pct"] < MEMORY_FREE_MIN_PCT:
        raise ResourceStop(f"free memory pressure fell below {MEMORY_FREE_MIN_PCT:.0f}%")
    if snap["swap"] is not None and snap["swap"]["fraction"] >= STOP_SWAP_MAX:
        raise ResourceStop(f"swap fraction {snap['swap']['fraction']:.4f} >= {STOP_SWAP_MAX:.4f}")
    return snap


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def task_hashes() -> dict[str, str]:
    return {task.task_id: task.sha256() for task in task_pack.all_tasks()}


def source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "scripts/qwen3_function_repair_grpo.py",
        ROOT / "scripts/long_horizon_eval.py",
        ROOT / "scripts/funcy_tree_oracle.py",
        ROOT / "scripts/run_local_agent.py",
        ROOT / "scripts/pinned_workspace_git.py",
        ROOT / "src/vare/grpo_mlx.py",
        ROOT / "src/vare/function_repair_tasks.py",
        ROOT / "src/vare/learning_pipeline/policy.py",
        ROOT / "src/vare/agent_edit_format.py",
        ROOT / "src/vare/trajectory_store.py",
        ROOT / "src/vare/eval_stats.py",
        ROOT / "src/vare/evidence.py",
        ROOT / "src/vare/isolated_candidate.py",
        ROOT / "src/vare/container_isolation.py",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def verifier_id(
    task: task_pack.FunctionRepairTask, *, image: str = CONTAINER_IMAGE,
) -> str:
    return sha256_json({
        "schema": "vare_parent_owned_function_oracle_v1",
        "task_id": task.task_id,
        "task_sha256": task.sha256(),
        "scorer_sha256": sha256_file(ROOT / "scripts/qwen3_function_repair_grpo.py"),
        "candidate_worker_sha256": sha256_file(ROOT / "src/vare/isolated_candidate.py"),
        "isolation_sha256": sha256_file(ROOT / "src/vare/container_isolation.py"),
        "container_image": image,
        "timeout_s": EVAL_TIMEOUT_SECONDS,
        "memory_mb": EVAL_MEMORY_MB,
        "all_cases_required": True,
    })


def task_prompt(task: task_pack.FunctionRepairTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt + "\n\nReturn only the function source now."},
    ]


def extract_source(text: str) -> tuple[str | None, str]:
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        matches = [block.strip() for block in blocks if re.search(r"\bdef\s+solve\s*\(", block)]
        if len(matches) != 1:
            return None, "code_block_missing_or_ambiguous"
        source = matches[0] + "\n"
    else:
        match = re.search(r"(?m)^\s*def\s+solve\s*\(", text)
        if match is None:
            return None, "solve_definition_missing"
        source = text[match.start():].strip() + "\n"
    if len(source.encode("utf-8")) > 128 * 1024:
        return None, "source_too_large"
    return source, "source_extracted"


def _json_value_matches(actual: Any, expected: Any) -> bool:
    """Compare decoded JSON values exactly, with numeric int/float equality."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        try:
            left, right = float(actual), float(expected)
        except OverflowError:
            return type(actual) is type(expected) is int and actual == expected
        return math.isfinite(left) and math.isfinite(right) and left == right
    if isinstance(actual, str) or isinstance(expected, str):
        return type(actual) is type(expected) is str and actual == expected
    if type(actual) is list or type(expected) is list:
        return (
            type(actual) is type(expected) is list
            and len(actual) == len(expected)
            and all(_json_value_matches(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if type(actual) is dict or type(expected) is dict:
        return (
            type(actual) is type(expected) is dict
            and set(actual) == set(expected)
            and all(_json_value_matches(actual[key], expected[key]) for key in actual)
        )
    return False


def score_source(
    task: task_pack.FunctionRepairTask, text: str, *, image: str = CONTAINER_IMAGE,
) -> dict[str, Any]:
    source, extraction = extract_source(text)
    if source is None:
        return {"reward": 0.0, "grade_status": extraction, "trusted_contract": True,
                "passed": False, "cases": []}
    with tempfile.TemporaryDirectory(prefix="vare-qwen3-function-oracle-") as temp:
        subject = Path(temp)
        (subject / "solution.py").write_text(source, encoding="utf-8")
        isolation = DockerIsolationRunner()
        case_results = []
        for case in task.cases:
            value, wall_s, failure = _one_call(
                subject, "solution", "solve", list(case.args), {},
                EVAL_TIMEOUT_SECONDS, image, isolation, (), EVAL_MEMORY_MB,
                True, None,
            )
            if failure is None and not _json_value_matches(value, case.expected):
                failure = "wrong_output"
            case_results.append({
                "case_id": case.case_id,
                "passed": failure is None,
                "failure": failure,
                "wall_s": wall_s,
            })
    passed = bool(case_results) and all(case["passed"] for case in case_results)
    return {
        "reward": 1.0 if passed else 0.0,
        "grade_status": "all_cases_passed" if passed else "case_failed",
        "trusted_contract": True,
        "passed": passed,
        "candidate_source": source,
        "candidate_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "case_set_sha256": verifier_id(task, image=image),
        "cases": case_results,
    }


def seed_for(base_seed: int, phase: str, task_id: str, sample_index: int) -> int:
    data = f"{base_seed}:{phase}:{task_id}:{sample_index}".encode("utf-8")
    return int(hashlib.sha256(data).hexdigest()[:8], 16)


def generate(model: Any, tokenizer: Any, messages: Sequence[Mapping[str, str]], *, seed: int,
             temperature: float, max_tokens: int) -> dict[str, Any]:
    if temperature > 0:
        grpo_mlx.check_exact_ratio_sampling(temperature=temperature, top_p=1.0)
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = list(tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=True))
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        return {"prompt_ids": prompt_ids, "completion_ids": [], "text": "", "finish": "prompt_too_long",
                "sampler_logprobs": [], "seconds": 0.0}
    mx.random.seed(seed)
    sampler = make_sampler(temp=temperature, top_p=1.0 if temperature > 0 else 0.0,
                           min_p=0.0, top_k=0)
    started = time.perf_counter()
    token_ids: list[int] = []
    sampler_logprobs: list[float] = []
    text = ""
    finish = "length"
    for item in stream_generate(model, tokenizer, prompt_ids, max_tokens=max_tokens, sampler=sampler):
        text += item.text
        token_ids.append(int(item.token))
        if getattr(item, "logprobs", None) is not None:
            sampler_logprobs.append(float(item.logprobs[item.token]))
        if item.finish_reason:
            finish = str(item.finish_reason)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": token_ids,
        "text": text,
        "finish": finish,
        "sampler_logprobs": sampler_logprobs,
        "seconds": round(time.perf_counter() - started, 6),
    }


def load_policy(*, seed: int, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model, tokenizer = load(model_path, adapter_path=str(adapter_path) if adapter_path else None)
    if adapter_path is None:
        grpo_mlx.make_lora_policy(model, **LORA_CONFIG, seed=seed)
        mx.eval(model.parameters())
    return model, tokenizer


@contextmanager
def gradient_checkpointing(model: Any):
    from mlx_lm.tuner.trainer import grad_checkpoint

    layer = model.layers[0]
    cls = type(layer)
    original = cls.__call__
    grad_checkpoint(layer)
    try:
        yield
    finally:
        cls.__call__ = original


def rng_probe(seed: int, update_step: int) -> list[float]:
    import mlx.core as mx

    grpo_mlx.reseed(seed, update_step)
    values = mx.random.uniform(shape=(8,), dtype=mx.float32)
    mx.eval(values)
    return [float(value) for value in values.tolist()]


def checkpoint_path(run: Path, name: str) -> Path:
    return run / "checkpoints" / name


def adapter_config() -> dict[str, Any]:
    return {
        "fine_tune_type": "lora",
        "num_layers": LORA_CONFIG["num_layers"],
        "lora_parameters": {"rank": LORA_CONFIG["rank"], "scale": LORA_CONFIG["scale"],
                            "dropout": LORA_CONFIG["dropout"]},
    }


def export_adapter(checkpoint: Path, destination: Path) -> str:
    """Export the trained weights in mlx-lm's standard inference adapter format."""
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite adapter export: {destination}")
    destination.mkdir(parents=True)
    shutil.copyfile(checkpoint / "adapters.safetensors", destination / "adapters.safetensors")
    atomic_json(destination / "adapter_config.json", adapter_config())
    return sha256_directory(destination)


def make_policy_id(model: Any, update_step: int) -> str:
    return f"qwen3-4b-mlx:{MODEL_REVISION[:12]}:u{update_step}:{grpo_mlx.trainable_digest(model)[:20]}"


def protocol_payload(path: Path, *, seed: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen experiment protocol is required before model execution: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sampling", {}).get("seed") != seed:
        raise RuntimeError("CLI seed differs from the frozen protocol seed")
    if payload.get("source_hashes") != source_hashes():
        raise RuntimeError("experiment source hashes differ from the frozen protocol")
    contracts = payload.get("task_contracts", {})
    contracted_tasks = {
        **contracts.get("train", {}).get("task_hashes", {}),
        **contracts.get("development", {}).get("task_hashes", {}),
    }
    if contracted_tasks != task_hashes():
        raise RuntimeError("task contracts differ from the frozen protocol")
    if not payload.get("preregistered_before_model_inference"):
        raise RuntimeError("model execution requires a preregistered protocol")
    return payload


def protocol_hash(path: Path = DEFAULT_PROTOCOL) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen experiment protocol is required before model execution: {path}")
    return sha256_file(path)


def run_identity(run: Path, phase: str, *, protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    meta_path = run / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError("run metadata is missing; start with phase=train")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_hashes") != source_hashes():
        raise RuntimeError("experiment source changed since the run was frozen")
    protocol = protocol_payload(protocol_path, seed=int(meta["seed"]))
    if meta.get("protocol_sha256") != protocol_hash(protocol_path):
        raise RuntimeError("protocol changed since the run was frozen")
    if meta.get("experiment_id") != protocol.get("planned_run_id"):
        raise RuntimeError("run identifier differs from the frozen protocol")
    if meta.get("task_hashes") != task_hashes():
        raise RuntimeError("task contracts changed since the run was frozen")
    return meta


def checkpoint_state(run: Path, path: Path, *, seed: int) -> tuple[dict[str, Any], Any, Any, Any]:
    import mlx.optimizers as optim

    model, tokenizer = load_policy(seed=seed)
    optimizer = optim.Adam(learning_rate=OPTIMIZER_CONFIG["learning_rate"])
    manifest = grpo_mlx.load_checkpoint(path, model, optimizer, base_model_id=BASE_MODEL_ID)
    if int(manifest["step"]) != int(optimizer.state["step"].item()):
        raise RuntimeError("checkpoint and optimizer update counts differ")
    return manifest, model, optimizer, tokenizer


def serialize_sample(sample: grpo_mlx.Sample, *, response: Mapping[str, Any], task_sha256: str,
                     phase: str) -> dict[str, Any]:
    return {
        "instance_id": sample.instance_id,
        "behavior_checkpoint_id": sample.behavior_checkpoint_id,
        "behavior_step": sample.behavior_step,
        "prompt_ids": sample.prompt_ids,
        "completion_ids": sample.completion_ids,
        "reward": sample.reward,
        "verifier": sample.verifier,
        "sample_index": sample.sample_index,
        "behavior_logprobs": sample.behavior_logprobs,
        "task_sha256": task_sha256,
        "phase": phase,
        "response_sha256": hashlib.sha256(response["text"].encode("utf-8")).hexdigest(),
        "candidate_source_sha256": response.get("candidate_source_sha256"),
        "finish": response.get("finish"),
    }


def deserialize_sample(row: Mapping[str, Any]) -> grpo_mlx.Sample:
    return grpo_mlx.Sample(
        instance_id=str(row["instance_id"]),
        behavior_checkpoint_id=str(row["behavior_checkpoint_id"]),
        behavior_step=int(row["behavior_step"]),
        prompt_ids=[int(token) for token in row["prompt_ids"]],
        completion_ids=[int(token) for token in row["completion_ids"]],
        reward=float(row["reward"]),
        verifier=str(row["verifier"]),
        sample_index=int(row["sample_index"]),
        behavior_logprobs=(None if row.get("behavior_logprobs") is None else
                           [float(value) for value in row["behavior_logprobs"]]),
        meta={"task_sha256": row["task_sha256"], "phase": row["phase"]},
    )


def informative_group_stats(samples: Sequence[grpo_mlx.Sample]) -> dict[str, Any]:
    groups = grpo_mlx.group_samples(samples, group_size=GROUP_SIZE)
    rows = []
    for (task_id, checkpoint_id), members in sorted(groups.items()):
        rewards = [member.reward for member in members]
        _, informative = grpo_mlx.group_advantages(rewards, normalize_std=True)
        rows.append({"task_id": task_id, "checkpoint_id": checkpoint_id, "rewards": rewards,
                     "informative": informative})
    return {"groups": rows, "n_groups": len(rows),
            "n_informative_groups": sum(row["informative"] for row in rows),
            "n_zero_variance_groups": sum(not row["informative"] for row in rows)}


def sample_tasks(model: Any, tokenizer: Any, *, tasks: Sequence[task_pack.FunctionRepairTask],
                 phase: str, update_step: int, policy_id: str, seed: int, max_tokens: int,
                 run: Path, phase_started: float) -> tuple[list[grpo_mlx.Sample], list[dict[str, Any]]]:
    samples: list[grpo_mlx.Sample] = []
    rows: list[dict[str, Any]] = []
    rollouts_path = run / f"rollouts-{phase}.jsonl"
    if rollouts_path.exists():
        raise FileExistsError(f"refusing to overwrite existing rollouts: {rollouts_path}")
    for task in tasks:
        if task.split != "train":
            raise ValueError("optimizer rollouts may use only Train tasks")
        for index in range(TRAIN_SAMPLES_PER_TASK):
            resource_guard(run, phase_started)
            sample_seed = seed_for(seed, phase, task.task_id, index)
            generated = generate(model, tokenizer, task_prompt(task), seed=sample_seed,
                                 temperature=GRPO_CONFIG.temperature, max_tokens=max_tokens)
            score = score_source(task, generated["text"])
            if generated["finish"] == "prompt_too_long":
                score = {"reward": 0.0, "grade_status": "prompt_too_long", "trusted_contract": True,
                         "passed": False, "cases": []}
            verifier = verifier_id(task)
            sample = grpo_mlx.Sample(
                instance_id=task.task_id, behavior_checkpoint_id=policy_id, behavior_step=update_step,
                prompt_ids=generated["prompt_ids"], completion_ids=generated["completion_ids"],
                reward=float(score["reward"]), verifier=verifier, sample_index=index,
                meta={"task_sha256": task.sha256(), "phase": phase},
            )
            response = {**generated, **score}
            row = {
                "schema": "vare_qwen3_function_repair_rollout_v1", "phase": phase,
                "task_id": task.task_id, "split": task.split, "task_sha256": task.sha256(),
                "policy_checkpoint_id": policy_id, "behavior_step": update_step,
                "sample_index": index, "seed": sample_seed,
                "temperature": GRPO_CONFIG.temperature, "top_p": 1.0,
                "sampling_distribution": "untruncated softmax(logits / temperature)",
                "prompt_tokens": len(generated["prompt_ids"]),
                "completion_tokens": len(generated["completion_ids"]),
                "generation_seconds": generated["seconds"], "finish": generated["finish"],
                "response_sha256": hashlib.sha256(generated["text"].encode("utf-8")).hexdigest(),
                "candidate_source_sha256": score.get("candidate_source_sha256"),
                "reward": score["reward"], "grade_status": score["grade_status"],
                "trusted_contract": score["trusted_contract"], "grader": {
                    "verifier_id": verifier, "case_results": score["cases"],
                    "case_set_sha256": score.get("case_set_sha256"),
                },
                "text": generated["text"],
            }
            append_jsonl(rollouts_path, row)
            samples.append(sample)
            rows.append({**serialize_sample(sample, response=response, task_sha256=task.sha256(), phase=phase),
                         "response_text": generated["text"], "sampler_logprobs": generated["sampler_logprobs"],
                         "reward_detail": {k: v for k, v in score.items() if k != "candidate_source"}})
            resource_guard(run, phase_started)
    return samples, rows


def development_evaluation(model: Any, tokenizer: Any, *, phase: str, seed: int,
                           run: Path, phase_started: float) -> list[dict[str, Any]]:
    path = run / f"development-{phase}.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Development results: {path}")
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(task_pack.DEVELOPMENT_TASKS):
        resource_guard(run, phase_started)
        generated = generate(model, tokenizer, task_prompt(task), seed=seed_for(seed, phase, task.task_id, index),
                             temperature=0.0, max_tokens=DEVELOPMENT_MAX_NEW_TOKENS)
        score = score_source(task, generated["text"])
        row = {
            "schema": "vare_qwen3_function_repair_development_v1", "phase": phase,
            "task_id": task.task_id, "split": task.split, "task_sha256": task.sha256(),
            "prompt_tokens": len(generated["prompt_ids"]),
            "completion_tokens": len(generated["completion_ids"]),
            "generation_seconds": generated["seconds"], "finish": generated["finish"],
            "response_sha256": hashlib.sha256(generated["text"].encode("utf-8")).hexdigest(),
            "candidate_source_sha256": score.get("candidate_source_sha256"),
            "reward": score["reward"], "grade_status": score["grade_status"],
            "trusted_contract": score["trusted_contract"], "case_results": score["cases"],
            "text": generated["text"],
        }
        append_jsonl(path, row)
        rows.append(row)
        resource_guard(run, phase_started)
    return rows


def rng_sentinel(seed: int, step: int) -> list[float]:
    import mlx.core as mx

    grpo_mlx.reseed(seed, step)
    values = mx.random.uniform(shape=(8,), dtype=mx.float32)
    mx.eval(values)
    return [float(value) for value in values.tolist()]


def write_checkpoint_state(run: Path, *, manifest: Mapping[str, Any], optimizer: Any,
                           model: Any, seed: int, applied_step: int, sentinel: list[float]) -> dict[str, Any]:
    checkpoint = {
        "schema": "vare_qwen3_function_repair_checkpoint_state_v1",
        "checkpoint": "checkpoints/update-0001",
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path(run, "update-0001") / "checkpoint.json"),
        "update_step": int(manifest["step"]),
        "optimizer_step": int(optimizer.state["step"].item()),
        "optimizer": OPTIMIZER_CONFIG,
        "optimizer_state_sha256": optimizer_digest(optimizer),
        "trainable_parameters_sha256": grpo_mlx.trainable_digest(model),
        "rng_base_seed": seed,
        "rng_step": applied_step,
        "rng_probe": sentinel,
        "rng_probe_sha256": sha256_json(sentinel),
        "files": manifest["files"],
    }
    atomic_json(run / "checkpoint-state.json", checkpoint)
    return checkpoint


def prepare_run(run: Path, *, seed: int, protocol_path: Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    protocol = protocol_payload(protocol_path, seed=seed)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run}")
    run.mkdir(parents=True, exist_ok=True)
    snap = preflight(run, "train")
    meta = {
        "schema": "vare_qwen3_function_repair_run_v1",
        "experiment_id": protocol["planned_run_id"],
        "starting_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                             capture_output=True, text=True).stdout.strip(),
        "source_hashes": source_hashes(),
        "task_hashes": task_hashes(),
        "protocol_path": str(protocol_path.expanduser().resolve().relative_to(ROOT)),
        "protocol_sha256": protocol_hash(protocol_path),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "python": sys.version,
        "platform": platform.platform(),
        "mlx_version": None,
        "mlx_lm_version": None,
        "seed": seed,
        "sampling": {"train_temperature": GRPO_CONFIG.temperature, "top_p": 1.0,
                     "top_k": 0, "min_p": 0.0,
                     "distribution": "untruncated softmax(logits / temperature)"},
        "grpo": GRPO_CONFIG.to_dict(), "lora": LORA_CONFIG,
        "optimizer": OPTIMIZER_CONFIG,
        "budgets": {"train_tasks": [task.task_id for task in task_pack.TRAIN_TASKS],
                    "samples_per_task": TRAIN_SAMPLES_PER_TASK, "group_size": GROUP_SIZE,
                    "train_max_new_tokens_each": TRAIN_MAX_NEW_TOKENS,
                    "development_tasks": [task.task_id for task in task_pack.DEVELOPMENT_TASKS],
                    "development_max_new_tokens_each": DEVELOPMENT_MAX_NEW_TOKENS,
                    "recovery_tasks": list(RECOVERY_TASK_IDS)},
        "verifier": {"image": CONTAINER_IMAGE, "timeout_s": EVAL_TIMEOUT_SECONDS,
                     "memory_mb": EVAL_MEMORY_MB, "all_cases_required": True},
        "preflight": snap,
        "claim_boundary": "Development-only function-repair result; not independent held-out improvement",
    }
    try:
        import mlx
        import mlx_lm

        meta["mlx_version"] = mlx.__version__
        meta["mlx_lm_version"] = mlx_lm.__version__
    except Exception:
        pass
    atomic_json(run / "run_meta.json", meta)
    atomic_json(run / "task_manifest.json", [
        {"task_id": task.task_id, "split": task.split, "task_sha256": task.sha256(),
         "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
         "case_count": len(task.cases)} for task in task_pack.all_tasks()
    ])
    return meta


def _maybe_sigkill(flag: bool, boundary: Mapping[str, Any]) -> None:
    if not flag:
        return
    print(json.dumps({"sigkill_boundary": boundary, "pid": os.getpid()}, sort_keys=True), flush=True)
    os.kill(os.getpid(), signal.SIGKILL)


def phase_train(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import mlx.optimizers as optim

    run = args.run_dir.expanduser().resolve()
    meta = prepare_run(run, seed=args.seed, protocol_path=args.protocol)
    phase_started = time.monotonic()
    phase_cpu_before = process_cpu_seconds()
    mx.reset_peak_memory()
    model, tokenizer = load_policy(seed=args.seed)
    resource_guard(run, phase_started)
    initial_digest = grpo_mlx.trainable_digest(model)
    policy_id = make_policy_id(model, 0)
    atomic_json(run / "base-policy.json", {"policy_checkpoint_id": policy_id,
                "trainable_parameters_sha256": initial_digest,
                "base_model_id": BASE_MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_path_sha256": sha256_json({"id": MODEL_ID, "revision": MODEL_REVISION})})
    dev_pre = development_evaluation(model, tokenizer, phase="pre", seed=args.seed,
                                     run=run, phase_started=phase_started)
    samples, sample_rows = sample_tasks(model, tokenizer, tasks=task_pack.TRAIN_TASKS, phase="train-step-0",
                                        update_step=0, policy_id=policy_id, seed=args.seed,
                                        max_tokens=TRAIN_MAX_NEW_TOKENS, run=run,
                                        phase_started=phase_started)
    groups = informative_group_stats(samples)
    atomic_json(run / "training-samples-before-logprob.json", sample_rows)
    generation_peak_mlx_gib = mx.get_peak_memory() / GIB
    optimizer = optim.Adam(learning_rate=OPTIMIZER_CONFIG["learning_rate"])
    if groups["n_informative_groups"] == 0:
        result = {"update_applied": False, "reason": "no_informative_group",
                  "update_step": 0, "groups": groups, "params_changed": False,
                  "development_pre": {"n": len(dev_pre), "passed": sum(row["reward"] == 1 for row in dev_pre)},
                  "resource": {"peak_mlx_gib": generation_peak_mlx_gib,
                               "peak_rss_gib": resource_snapshot(run)["peak_rss_gib"],
                               "cpu_seconds": process_cpu_seconds() - phase_cpu_before,
                               "wall_seconds": time.monotonic() - phase_started,
                               "end_snapshot": resource_snapshot(run)}}
        atomic_json(run / "training-result.json", result)
        return 0

    informative = grpo_mlx.informative_samples(samples, GRPO_CONFIG)
    mx.clear_cache()
    mx.reset_peak_memory()
    cpu_before = process_cpu_seconds()
    with gradient_checkpointing(model):
        grpo_mlx.compute_behavior_logprobs(model, informative, GRPO_CONFIG,
                                          after_sample=lambda: resource_guard(run, phase_started))
        behavior_done = time.monotonic()
        update = grpo_mlx.train_step(model, optimizer, samples, GRPO_CONFIG,
                                     policy_checkpoint_id=policy_id, policy_step=0,
                                     after_sample=lambda: resource_guard(run, phase_started))
    mx.eval(model.parameters(), optimizer.state)
    update_done = time.monotonic()
    if not update.get("update_applied") or not update.get("params_changed"):
        raise RuntimeError("informative group did not produce a changed LoRA update")
    if int(optimizer.state["step"].item()) != 1:
        raise RuntimeError("optimizer update count is not one after the first applied step")
    atomic_json(run / "training-samples.json", [
        serialize_sample(sample, response={"text": row["response_text"],
                                           "candidate_source_sha256": row["candidate_source_sha256"]},
                         task_sha256=row["task_sha256"], phase=row["phase"])
        for sample, row in zip(samples, sample_rows, strict=True)
    ])
    mx.clear_cache()
    update["resource"] = {"peak_mlx_gib": max(generation_peak_mlx_gib, mx.get_peak_memory() / GIB),
                          "peak_mlx_generation_gib": generation_peak_mlx_gib,
                          "peak_mlx_update_gib": mx.get_peak_memory() / GIB,
                          "peak_rss_gib": resource_snapshot(run)["peak_rss_gib"],
                          "cpu_seconds": process_cpu_seconds() - cpu_before,
                          "behavior_seconds": behavior_done - phase_started,
                          "update_seconds": update_done - behavior_done,
                          "wall_seconds": update_done - phase_started,
                          "phase_cpu_seconds": process_cpu_seconds() - phase_cpu_before,
                          "end_snapshot": resource_snapshot(run)}
    update["groups"] = groups
    update["training_task_count"] = len(task_pack.TRAIN_TASKS)
    update["generation_count"] = len(samples)
    ckpt = checkpoint_path(run, "update-0001")
    manifest = grpo_mlx.save_checkpoint(
        ckpt, model, optimizer, base_model_id=BASE_MODEL_ID, step=1, base_seed=args.seed,
        extra={"experiment_id": meta["experiment_id"], "model_revision": MODEL_REVISION,
               "policy_checkpoint_id_before": policy_id, "task_hashes": meta["task_hashes"],
               "optimizer": OPTIMIZER_CONFIG, "lora": LORA_CONFIG,
               "attempt_index": 1, "informative_groups": groups["n_informative_groups"]},
    )
    sentinel = rng_sentinel(args.seed, 1)
    ckpt_state = write_checkpoint_state(run, manifest=manifest, optimizer=optimizer,
                                        model=model, seed=args.seed, applied_step=1, sentinel=sentinel)
    update["checkpoint"] = ckpt_state
    update["params_digest_after"] = grpo_mlx.trainable_digest(model)
    update["optimizer_step_after"] = int(optimizer.state["step"].item())
    update["development_pre"] = {"n": len(dev_pre), "passed": sum(row["reward"] == 1 for row in dev_pre)}
    atomic_json(run / "training-result.json", update)
    atomic_json(run / "learner-events.json", [{"event": "optimizer_update_committed", "update_step": 1,
                "checkpoint_state_sha256": sha256_file(run / "checkpoint-state.json"),
                "parameter_sha256": update["params_digest_after"],
                "optimizer_sha256": ckpt_state["optimizer_state_sha256"],
                "time": time.time()}])
    print(json.dumps({"update_applied": True, "update_step": 1,
                      "informative_groups": groups["n_informative_groups"],
                      "parameter_sha256": update["params_digest_after"],
                      "optimizer_sha256": ckpt_state["optimizer_state_sha256"],
                      "checkpoint": str(ckpt)}, sort_keys=True), flush=True)
    _maybe_sigkill(args.sigkill_after_update, {"type": "after_optimizer_update_and_checkpoint",
                                                "update_step": 1,
                                                "checkpoint_state_sha256": sha256_file(run / "checkpoint-state.json")})
    return 0


def restore_and_verify(run: Path, *, seed: int, phase: str,
                       protocol_path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], Any, Any, dict[str, Any], Any]:
    preflight(run, phase)
    meta = run_identity(run, phase, protocol_path=protocol_path)
    if int(meta.get("seed", -1)) != seed:
        raise RuntimeError("run seed differs from the frozen experiment seed")
    checkpoint = checkpoint_path(run, "update-0001")
    manifest, model, optimizer, tokenizer = checkpoint_state(run, checkpoint, seed=seed)
    saved_state = json.loads((run / "checkpoint-state.json").read_text(encoding="utf-8"))
    checks = {
        "optimizer_step_matches_checkpoint": int(optimizer.state["step"].item()) == saved_state["update_step"],
        "parameter_digest_matches_checkpoint": grpo_mlx.trainable_digest(model) == saved_state[
            "trainable_parameters_sha256"],
        "optimizer_digest_matches_checkpoint": optimizer_digest(optimizer) == saved_state["optimizer_state_sha256"],
        "rng_probe_matches_checkpoint": rng_probe(seed, int(manifest["step"])) == saved_state["rng_probe"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"learner state recovery check failed: {checks}")
    atomic_json(run / f"recovery-restore-{phase}.json", {
        "phase": phase, "checkpoint_step": manifest["step"], "checks": checks,
        "parameter_sha256": grpo_mlx.trainable_digest(model),
        "optimizer_sha256": optimizer_digest(optimizer),
        "optimizer_step": int(optimizer.state["step"].item()),
        "rng_probe_sha256": sha256_json(saved_state["rng_probe"]),
        "model_revision": MODEL_REVISION, "time": time.time(),
    })
    return meta, model, optimizer, saved_state, tokenizer


def phase_prepare_recovery(args: argparse.Namespace) -> int:
    import mlx.core as mx

    run = args.run_dir.expanduser().resolve()
    meta, model, optimizer, saved_state, tokenizer = restore_and_verify(
        run, seed=args.seed, phase="prepare-recovery", protocol_path=args.protocol)
    phase_started = time.monotonic()
    current_policy_id = make_policy_id(model, 1)
    recovery_tasks = [task for task in task_pack.TRAIN_TASKS if task.task_id in RECOVERY_TASK_IDS]
    samples, sample_rows = sample_tasks(model, tokenizer, tasks=recovery_tasks,
                                        phase="train-step-1", update_step=1, policy_id=current_policy_id,
                                        seed=args.seed, max_tokens=TRAIN_MAX_NEW_TOKENS, run=run,
                                        phase_started=phase_started)
    groups = informative_group_stats(samples)
    informative = grpo_mlx.informative_samples(samples, GRPO_CONFIG)
    if informative:
        with gradient_checkpointing(model):
            grpo_mlx.compute_behavior_logprobs(model, informative, GRPO_CONFIG,
                                              after_sample=lambda: resource_guard(run, phase_started))
    sample_rows_by_key = {(row["instance_id"], row["sample_index"]): row for row in sample_rows}
    serialized = []
    for sample in samples:
        row = sample_rows_by_key[(sample.instance_id, sample.sample_index)]
        serialized.append({**serialize_sample(sample, response={"text": row["response_text"],
                                                                  "candidate_source_sha256": row[
                                                                      "candidate_source_sha256"]},
                                               task_sha256=sample.meta["task_sha256"], phase="train-step-1"),
                           "response_text": row["response_text"],
                           "sampler_logprobs": row["sampler_logprobs"],
                           "reward_detail": row["reward_detail"]})
    payload = {
        "schema": "vare_qwen3_function_repair_pending_update_v1",
        "starting_checkpoint_sha256": saved_state["checkpoint_manifest_sha256"],
        "starting_update_step": 1,
        "behavior_checkpoint_id": current_policy_id,
        "policy_parameters_sha256": grpo_mlx.trainable_digest(model),
        "optimizer_state_sha256": optimizer_digest(optimizer),
        "task_hashes": {task.task_id: task.sha256() for task in recovery_tasks},
        "groups": groups,
        "samples": serialized,
        "seed": args.seed,
        "created_at": time.time(),
    }
    payload["payload_sha256"] = sha256_json(payload)
    atomic_json(run / "pending-update-0002.json", payload)
    atomic_json(run / "recovery-boundary-pending.json", {
        "boundary": "after_rollout_rewards_and_behavior_logprobs_committed_before_update",
        "pending_sha256": payload["payload_sha256"], "samples": len(samples),
        "informative_groups": groups["n_informative_groups"],
        "optimizer_step": int(optimizer.state["step"].item()),
        "parameter_sha256": grpo_mlx.trainable_digest(model), "time": time.time(),
    })
    mx.clear_cache()
    print(json.dumps({"pending_samples": len(samples), "informative_groups": groups["n_informative_groups"],
                      "pending_sha256": payload["payload_sha256"]}, sort_keys=True), flush=True)
    _maybe_sigkill(args.sigkill_after_pending, {"type": "after_pending_rollouts_committed_before_update",
                                                 "pending_sha256": payload["payload_sha256"]})
    return 0


def train_pending_on_checkpoint(run: Path, *, seed: int, checkpoint_name: str,
                                pending: Sequence[Mapping[str, Any]], branch: str) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim

    preflight(run, branch)
    phase_started = time.monotonic()
    checkpoint = checkpoint_path(run, "update-0001")
    model, tokenizer = load_policy(seed=seed)
    resource_guard(run, phase_started)
    optimizer = optim.Adam(learning_rate=OPTIMIZER_CONFIG["learning_rate"])
    manifest = grpo_mlx.load_checkpoint(checkpoint, model, optimizer, base_model_id=BASE_MODEL_ID)
    if manifest["step"] != 1 or int(optimizer.state["step"].item()) != 1:
        raise RuntimeError("pending update must resume from optimizer step 1")
    samples = [deserialize_sample(row) for row in pending]
    pending_groups = informative_group_stats(samples)
    before_digest = grpo_mlx.trainable_digest(model)
    before_optimizer_digest = optimizer_digest(optimizer)
    if any(sample.behavior_checkpoint_id != make_policy_id(model, 1) or sample.behavior_step != 1
           for sample in samples):
        raise RuntimeError("pending rollouts are stale or belong to another policy checkpoint")
    with gradient_checkpointing(model):
        update = grpo_mlx.train_step(
            model, optimizer, samples, GRPO_CONFIG,
            policy_checkpoint_id=make_policy_id(model, 1), policy_step=1,
            after_sample=lambda: resource_guard(run, phase_started),
        )
    mx.eval(model.parameters(), optimizer.state)
    if update["update_applied"]:
        update_manifest = grpo_mlx.save_checkpoint(
            checkpoint_path(run, checkpoint_name), model, optimizer, base_model_id=BASE_MODEL_ID,
            step=2, base_seed=seed,
            extra={"experiment_id": json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["experiment_id"],
                   "starting_checkpoint": "checkpoints/update-0001",
                   "pending_update_sha256": sha256_json(list(pending)),
                   "optimizer": OPTIMIZER_CONFIG, "branch": branch},
        )
    else:
        update_manifest = None
    result = {
        "branch": branch,
        "groups": pending_groups,
        "update": update,
        "starting_parameter_sha256": before_digest,
        "starting_optimizer_sha256": before_optimizer_digest,
        "ending_parameter_sha256": grpo_mlx.trainable_digest(model),
        "ending_optimizer_sha256": optimizer_digest(optimizer),
        "optimizer_step": int(optimizer.state["step"].item()),
        "checkpoint_manifest": update_manifest,
        "peak_mlx_gib": mx.get_peak_memory() / GIB,
        "process_cpu_seconds": process_cpu_seconds(),
    }
    del tokenizer
    return result


def phase_resume(args: argparse.Namespace) -> int:
    run = args.run_dir.expanduser().resolve()
    meta, model, optimizer, saved_state, _tokenizer = restore_and_verify(
        run, seed=args.seed, phase="resume", protocol_path=args.protocol)
    del model, optimizer, _tokenizer
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    pending_path = run / "pending-update-0002.json"
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    recorded_digest = payload.pop("payload_sha256", None)
    if recorded_digest != sha256_json(payload):
        raise RuntimeError("pending rollout record hash mismatch")
    if payload.get("starting_checkpoint_sha256") != saved_state["checkpoint_manifest_sha256"]:
        raise RuntimeError("pending rollouts were generated from another checkpoint")
    if payload.get("policy_parameters_sha256") != saved_state["trainable_parameters_sha256"]:
        raise RuntimeError("pending rollouts have a different policy parameter identity")
    if payload.get("optimizer_state_sha256") != saved_state["optimizer_state_sha256"]:
        raise RuntimeError("pending rollouts have a different optimizer identity")
    rows = payload["samples"]
    recovered = train_pending_on_checkpoint(run, seed=args.seed, checkpoint_name="recovered-update-0002",
                                            pending=rows, branch="sigkill-resume")
    atomic_json(run / "recovered-update-0002.json", recovered)
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    replay = train_pending_on_checkpoint(run, seed=args.seed, checkpoint_name="replay-control-update-0002",
                                         pending=rows, branch="checkpoint-replay-control")
    comparison = {
        "schema": "vare_qwen3_checkpoint_replay_comparison_v1",
        "starting_checkpoint_sha256": saved_state["checkpoint_manifest_sha256"],
        "pending_sha256": recorded_digest,
        "recovered_parameter_sha256": recovered["ending_parameter_sha256"],
        "replay_parameter_sha256": replay["ending_parameter_sha256"],
        "recovered_optimizer_sha256": recovered["ending_optimizer_sha256"],
        "replay_optimizer_sha256": replay["ending_optimizer_sha256"],
        "recovered_optimizer_step": recovered["optimizer_step"],
        "replay_optimizer_step": replay["optimizer_step"],
        "parameters_exact_match": recovered["ending_parameter_sha256"] == replay["ending_parameter_sha256"],
        "optimizer_state_exact_match": recovered["ending_optimizer_sha256"] == replay["ending_optimizer_sha256"],
        "update_applied_both": bool(recovered["update"]["update_applied"] and replay["update"]["update_applied"]),
    }
    atomic_json(run / "checkpoint-replay-comparison.json", comparison)

    import mlx.core as mx
    import mlx.optimizers as optim
    model, tokenizer = load_policy(seed=args.seed)
    post_optimizer = optim.Adam(learning_rate=OPTIMIZER_CONFIG["learning_rate"])
    if recovered["update"]["update_applied"]:
        selected_checkpoint = checkpoint_path(run, "recovered-update-0002")
    else:
        selected_checkpoint = checkpoint_path(run, "update-0001")
    grpo_mlx.load_checkpoint(selected_checkpoint, model, post_optimizer, base_model_id=BASE_MODEL_ID)
    adapter = run / "adapter-export"
    adapter_digest = export_adapter(selected_checkpoint, adapter)
    phase_started = time.monotonic()
    post = development_evaluation(model, tokenizer, phase="post", seed=args.seed,
                                  run=run, phase_started=phase_started)
    adapter_check_task = task_pack.DEVELOPMENT_TASKS[0]
    reference_row = next(row for row in post if row["task_id"] == adapter_check_task.task_id)
    reference_hash = reference_row["response_sha256"]
    del model, tokenizer, post_optimizer
    gc.collect()
    mx.clear_cache()
    reload_started = time.monotonic()
    reload_model, reload_tokenizer = load_policy(seed=args.seed, adapter_path=adapter)
    resource_guard(run, reload_started)
    reloaded_generation = generate(
        reload_model, reload_tokenizer, task_prompt(adapter_check_task),
        seed=seed_for(args.seed, "post", adapter_check_task.task_id, 0),
        temperature=0.0, max_tokens=DEVELOPMENT_MAX_NEW_TOKENS,
    )
    reloaded_hash = hashlib.sha256(reloaded_generation["text"].encode("utf-8")).hexdigest()
    adapter_reload_check = {
        "adapter_path": str(adapter),
        "adapter_directory_sha256": adapter_digest,
        "checkpoint_adapter_sha256": sha256_file(selected_checkpoint / "adapters.safetensors"),
        "development_task_id": adapter_check_task.task_id,
        "checkpoint_loaded_response_sha256": reference_hash,
        "adapter_loaded_response_sha256": reloaded_hash,
        "response_exact_match": reference_hash == reloaded_hash,
        "generation_seconds": reloaded_generation["seconds"],
    }
    del reload_model, reload_tokenizer
    gc.collect()
    mx.clear_cache()
    result = {
        "schema": "vare_qwen3_function_repair_final_v1",
        "training_update_1": json.loads((run / "training-result.json").read_text(encoding="utf-8")),
        "checkpoint_replay": comparison,
        "development_post": {"n": len(post), "passed": sum(row["reward"] == 1 for row in post),
                              "task_results": [{"task_id": row["task_id"], "reward": row["reward"],
                                                "response_sha256": row["response_sha256"]} for row in post]},
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_digest,
        "adapter_reload_check": adapter_reload_check,
        "model_revision": MODEL_REVISION,
    }
    pre_rows = [json.loads(line) for line in (run / "development-pre.jsonl").read_text(encoding="utf-8").splitlines()]
    result["development_pre"] = {
        "n": len(pre_rows), "passed": sum(row["reward"] == 1 for row in pre_rows),
        "task_results": [{"task_id": row["task_id"], "reward": row["reward"],
                          "response_sha256": row["response_sha256"]} for row in pre_rows],
    }
    result["resource_end"] = resource_snapshot(run)
    atomic_json(run / "final-summary.json", result)
    atomic_json(run / "postflight.json", {"resource": result["resource_end"], "adapter": str(adapter),
                                           "adapter_sha256": result["adapter_sha256"],
                                           "training_update_count": recovered["optimizer_step"]})
    print(json.dumps({"development_pre": result["development_pre"],
                      "development_post": result["development_post"],
                      "checkpoint_replay": comparison,
                      "adapter_sha256": result["adapter_sha256"]}, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="phase", required=True)
    train = sub.add_parser("train", help="baseline Development, Train rollouts, and first GRPO update")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    train.add_argument("--seed", type=int, default=20261101)
    train.add_argument("--sigkill-after-update", action="store_true",
                       help="send SIGKILL after the first optimizer update and its checkpoint are durable")
    prepare = sub.add_parser("prepare-recovery", help="restore update 1 and durably prepare fresh on-policy rollouts")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare.add_argument("--seed", type=int, default=20261101)
    prepare.add_argument("--sigkill-after-pending", action="store_true",
                          help="send SIGKILL after the next rollout batch and behavior log-probs are durable")
    resume = sub.add_parser("resume", help="restore pending update and replay it from the same checkpoint")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    resume.add_argument("--seed", type=int, default=20261101)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.phase == "train":
            return phase_train(args)
        if args.phase == "prepare-recovery":
            return phase_prepare_recovery(args)
        return phase_resume(args)
    except ResourceStop as exc:
        run = args.run_dir.expanduser().resolve()
        detail = {"phase": args.phase, "reason": str(exc),
                  "resource": resource_snapshot(run), "time": time.time()}
        atomic_json(run / f"resource-stop-{args.phase}.json", detail)
        print(json.dumps({"resource_stop": detail}, sort_keys=True), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
