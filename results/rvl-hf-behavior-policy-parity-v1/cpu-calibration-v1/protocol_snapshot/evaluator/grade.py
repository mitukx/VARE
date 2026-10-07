#!/usr/bin/env python3
"""CPU-only independent grader for the pinned HF behavior-policy task."""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


TRAINER_RELATIVE = Path("src/rvl_systems/hf_trainer.py")

FIXTURE_CASES = (
    {
        "case_id": "repeat-positive-logit",
        "prompt": "parity-case-positive",
        "prompt_ids": [1, 2],
        "response_ids": [2, 3, 9],
        "temperature": 0.7,
        "logits": [
            [0.4, 0.2, 2.0, 0.8, -0.3, -0.1, 0.4, 0.0, -0.4, 0.1],
            [0.1, 0.3, 1.1, 1.2, -0.5, 0.4, 0.2, 0.0, -0.2, 0.5],
            [0.2, -0.1, 1.4, 0.2, 0.3, 0.0, -0.4, 0.1, -0.2, 0.9],
        ],
    },
    {
        "case_id": "repeat-negative-logit",
        "prompt": "parity-case-negative",
        "prompt_ids": [4, 5],
        "response_ids": [5, 6, 9],
        "temperature": 1.3,
        "logits": [
            [0.2, -0.3, 0.6, 0.1, -0.5, -1.1, 0.8, 0.4, -0.2, 0.0],
            [0.1, 0.2, -0.4, 0.0, 0.7, 1.0, -0.8, 0.3, 0.5, 0.6],
            [0.4, -0.1, 0.0, 0.2, -0.5, 0.8, 0.1, -0.2, 0.3, 1.1],
        ],
    },
    {
        "case_id": "repeated-context-temperature-one",
        "prompt": "parity-case-third",
        "prompt_ids": [7, 3],
        "response_ids": [3, 8, 9],
        "temperature": 1.0,
        "logits": [
            [-0.2, 0.3, 0.1, 1.7, 0.4, -0.1, 0.0, 0.6, 0.5, 0.2],
            [0.1, 0.0, -0.3, 0.8, 0.2, 0.4, -0.2, 0.7, 1.2, 0.3],
            [0.2, -0.1, 0.4, 1.1, -0.2, 0.3, 0.0, -0.4, 0.1, 0.8],
        ],
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _log_softmax(values: list[float]) -> list[float]:
    peak = max(values)
    log_normalizer = peak + math.log(sum(math.exp(value - peak) for value in values))
    return [value - log_normalizer for value in values]


class _Tensor:
    """Small list-backed tensor surface used only by the deterministic fixture."""

    def __init__(self, values: Any):
        self.values = values

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.values, (list, tuple)):
            if self.values and isinstance(self.values[0], (list, tuple)):
                return (len(self.values), len(self.values[0]))
            return (len(self.values),)
        return ()

    def __getitem__(self, key: Any) -> "_Tensor":
        value = self.values
        for index in key if isinstance(key, tuple) else (key,):
            value = value[index]
        return _Tensor(value)

    def to(self, _device: str) -> "_Tensor":
        return self

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def float(self) -> "_Tensor":
        return self

    def tolist(self) -> Any:
        if isinstance(self.values, tuple):
            return list(self.values)
        if isinstance(self.values, list):
            return [value.tolist() if isinstance(value, _Tensor) else value for value in self.values]
        return self.values


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self, cases: tuple[dict[str, Any], ...]):
        self._by_prompt = {case["prompt"]: case for case in cases}

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, _Tensor]:
        if return_tensors != "pt" or prompt not in self._by_prompt:
            raise ValueError("unexpected fixture tokenizer request")
        return {"input_ids": _Tensor([self._by_prompt[prompt]["prompt_ids"]])}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        kept = [token for token in token_ids if not (skip_special_tokens and token == self.eos_token_id)]
        return " ".join(str(token) for token in kept)


class _GenerationConfig:
    _DEFAULTS = {
        "_from_model_config": False,
        "_commit_hash": None,
        "transformers_version": "fixture",
        "max_length": 20,
        "max_new_tokens": None,
        "min_length": 0,
        "min_new_tokens": None,
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "num_return_sequences": 1,
        "length_penalty": 1.0,
        "diversity_penalty": 0.0,
        "use_cache": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "renormalize_logits": False,
        "constraints": None,
        "force_words_ids": None,
        "sequence_bias": None,
        "token_healing": False,
        "guidance_scale": None,
        "low_memory": None,
        "watermarking_config": None,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
        "output_scores": False,
        "return_dict_in_generate": False,
    }

    def __init__(self, **kwargs: Any):
        self._values = dict(self._DEFAULTS)
        self._values.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._values)


class _Model:
    def __init__(self, cases: tuple[dict[str, Any], ...], *, fail: bool = False):
        self.training = True
        self.generation_config = types.SimpleNamespace(
            bos_token_id=None,
            eos_token_id=9,
            pad_token_id=0,
            do_sample=True,
            temperature=0.8,
            top_k=37,
            top_p=0.83,
            repetition_penalty=1.7,
            no_repeat_ngram_size=0,
            suppress_tokens=None,
        )
        self._by_prompt_ids = {tuple(case["prompt_ids"]): case for case in cases}
        self._fail = fail
        self.effective_settings: dict[str, Any] = {}
        self._transition: list[float] = []

    def eval(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def generate(self, *, input_ids: _Tensor, **kwargs: Any) -> Any:
        if self._fail:
            raise RuntimeError("fixture generation failure")
        prompt_ids = tuple(input_ids.tolist()[0])
        case = self._by_prompt_ids[prompt_ids]
        repetition_penalty = float(
            kwargs.get("repetition_penalty", self.generation_config.repetition_penalty)
        )
        do_sample = bool(kwargs.get("do_sample", self.generation_config.do_sample))
        temperature = float(kwargs.get("temperature", self.generation_config.temperature))
        if not do_sample:
            temperature = 1.0
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("fixture got invalid generation temperature")
        self.effective_settings = {
            "repetition_penalty": repetition_penalty,
            "temperature": temperature,
            "top_k": kwargs.get("top_k", self.generation_config.top_k),
            "top_p": kwargs.get("top_p", self.generation_config.top_p),
            "no_repeat_ngram_size": kwargs.get(
                "no_repeat_ngram_size", self.generation_config.no_repeat_ngram_size
            ),
            "suppress_tokens": kwargs.get("suppress_tokens", self.generation_config.suppress_tokens),
        }
        history = list(prompt_ids)
        self._transition = []
        for raw_logits, token_id in zip(case["logits"], case["response_ids"]):
            adjusted = list(raw_logits)
            for seen_id in set(history):
                score = adjusted[seen_id]
                adjusted[seen_id] = score * repetition_penalty if score < 0 else score / repetition_penalty
            scaled = [score / temperature for score in adjusted]
            self._transition.append(_log_softmax(scaled)[token_id])
            history.append(token_id)
        sequence = list(prompt_ids) + list(case["response_ids"])
        return types.SimpleNamespace(sequences=_Tensor([sequence]), scores=[])

    def compute_transition_scores(
        self, _sequences: _Tensor, _scores: list[Any], *, normalize_logits: bool
    ) -> _Tensor:
        if not normalize_logits:
            raise ValueError("fixture requires normalized transition scores")
        return _Tensor([self._transition])


class _Torch:
    @staticmethod
    def manual_seed(_seed: int) -> None:
        return None

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()


def _install_fixture_modules() -> _Torch:
    torch_module = types.ModuleType("torch")
    torch_module.manual_seed = _Torch.manual_seed
    torch_module.inference_mode = _Torch.inference_mode
    transformers_module = types.ModuleType("transformers")
    transformers_module.GenerationConfig = _GenerationConfig
    sys.modules["torch"] = torch_module
    sys.modules["transformers"] = transformers_module
    return _Torch()


def _load_backend(source_root: Path):
    package_path = source_root / "src/rvl_systems"
    if not package_path.is_dir():
        raise FileNotFoundError(f"missing source package: {package_path}")
    for name in tuple(sys.modules):
        if name == "rvl_systems" or name.startswith("rvl_systems."):
            del sys.modules[name]
    package = types.ModuleType("rvl_systems")
    package.__path__ = [str(package_path)]
    sys.modules["rvl_systems"] = package
    return importlib.import_module("rvl_systems.hf_backend").HFLocalBackend


def _trainer_contract(path: Path) -> dict[str, bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    owner = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HFCausalLMGRPOTrainer"),
        None,
    )
    method = next(
        (node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "_sample_objective"),
        None,
    ) if owner else None
    if method is None:
        return {"trainer_method_present": False, "temperature_loaded": False,
                "temperature_scales_logits": False, "greedy_policy_rejected": False}

    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
    temperature_loaded = any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "sampling_temperature"
        for node in calls
    )
    temperature_scales_logits = any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "log_softmax"
        and node.args
        and ast.unparse(node.args[0]) == "response_logits.float() / temperature"
        for node in calls
    )
    rejects_greedy = (
        "temperature <= 0" in ast.unparse(method)
        and "greedy scores are not behavior probabilities" in ast.unparse(method)
    )
    return {
        "trainer_method_present": True,
        "temperature_loaded": temperature_loaded,
        "temperature_scales_logits": temperature_scales_logits,
        "greedy_policy_rejected": rejects_greedy,
    }


def _grade_rollouts(source_root: Path, acceptance: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch = _install_fixture_modules()
    backend_type = _load_backend(source_root)
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in FIXTURE_CASES:
        for temperature in (case["temperature"], 1.0):
            tokenizer = _Tokenizer((case,))
            model = _Model((case,))
            backend = backend_type("fixture-only", max_new_tokens=8, device="cpu")
            backend._torch = torch
            backend._tokenizer = tokenizer
            backend._model = model
            backend._device = "cpu"
            backend._resolved_precision = "fp32"
            generations = backend._generate_sync(
                case["case_id"], case["prompt"], n=1, temperature=temperature, seed=17
            )
            generation = generations[0]
            actual = generation.metadata.get("response_token_logprobs")
            response_ids = generation.metadata.get("response_token_ids")
            expected_ids = case["response_ids"]
            expected = [
                _log_softmax([value / temperature for value in row])[token_id]
                for row, token_id in zip(case["logits"], expected_ids)
            ]
            if response_ids != expected_ids or not isinstance(actual, list) or len(actual) != len(expected):
                error = None
            else:
                error = max(abs(float(left) - right) for left, right in zip(actual, expected))
            recorded_temperature = generation.metadata.get("sampling_temperature")
            row = {
                "case_id": case["case_id"],
                "temperature": temperature,
                "effective_repetition_penalty": model.effective_settings.get("repetition_penalty"),
                "response_token_ids": response_ids,
                "expected_logprobs": expected,
                "observed_logprobs": actual,
                "max_abs_logprob_error": error,
                "recorded_temperature": recorded_temperature,
                "model_training_state_restored": model.training,
            }
            outputs.append(row)
            if error is None or error > float(acceptance["max_abs_logprob_error"]):
                failures.append({"check": "rollout_probability_parity", "case_id": case["case_id"],
                                "temperature": temperature, "observed_error": error})
            if recorded_temperature is None or not math.isclose(float(recorded_temperature), temperature):
                failures.append({"check": "sampling_temperature_provenance", "case_id": case["case_id"],
                                "temperature": temperature, "observed": recorded_temperature})
            if model.effective_settings.get("repetition_penalty") != acceptance["neutral_repetition_penalty"]:
                failures.append({"check": "neutral_repetition_penalty", "case_id": case["case_id"],
                                "observed": model.effective_settings.get("repetition_penalty")})
            if (model.effective_settings.get("top_k") != acceptance["neutral_top_k"]
                    or model.effective_settings.get("top_p") != acceptance["neutral_top_p"]):
                failures.append({"check": "neutral_sampling_truncation", "case_id": case["case_id"],
                                "top_k": model.effective_settings.get("top_k"),
                                "top_p": model.effective_settings.get("top_p")})
            if not model.training:
                failures.append({"check": "model_state_restoration", "case_id": case["case_id"]})
    return outputs, failures


def _grade_error_restoration(source_root: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    case = FIXTURE_CASES[0]
    torch = _install_fixture_modules()
    backend_type = _load_backend(source_root)
    model = _Model((case,), fail=True)
    backend = backend_type("fixture-only", max_new_tokens=8, device="cpu")
    backend._torch = torch
    backend._tokenizer = _Tokenizer((case,))
    backend._model = model
    backend._device = "cpu"
    backend._resolved_precision = "fp32"
    raised = False
    try:
        backend._generate_sync("raises", case["prompt"], n=1, temperature=1.0, seed=17)
    except RuntimeError as exc:
        raised = str(exc) == "fixture generation failure"
    restored = model.training is True
    if not raised:
        failures.append({"check": "generation_exception_propagated"})
    if not restored:
        failures.append({"check": "model_state_restoration_after_error"})
    return {"expected_exception_propagated": raised, "model_training_state_restored": restored}


def _grade_invalid_temperatures(source_root: Path, failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case = FIXTURE_CASES[0]
    torch = _install_fixture_modules()
    backend_type = _load_backend(source_root)
    outcomes = []
    for temperature in (float("nan"), float("inf")):
        model = _Model((case,))
        backend = backend_type("fixture-only", max_new_tokens=8, device="cpu")
        backend._torch = torch
        backend._tokenizer = _Tokenizer((case,))
        backend._model = model
        backend._device = "cpu"
        backend._resolved_precision = "fp32"
        rejected = False
        try:
            backend._generate_sync("invalid", case["prompt"], n=1, temperature=temperature, seed=17)
        except ValueError:
            rejected = True
        outcome = {"temperature": str(temperature), "rejected": rejected,
                   "model_training_state_restored": model.training}
        outcomes.append(outcome)
        if not rejected:
            failures.append({"check": "invalid_temperature_rejected", "temperature": str(temperature)})
        if not model.training:
            failures.append({"check": "model_state_restoration_after_invalid_input",
                            "temperature": str(temperature)})
    return outcomes


def grade(workspace: Path, task_root: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    task_root = task_root.resolve()
    evaluator_file = Path(__file__).resolve()
    if evaluator_file.is_relative_to(workspace):
        raise ValueError("evaluator must be outside the candidate workspace")
    if task_root == workspace or task_root.is_relative_to(workspace):
        raise ValueError("task specification must be outside the candidate workspace")
    task = _json(task_root / "task.json")
    lock = _json(task_root / "protocol.lock.json")
    locked_hashes = lock["locked_hashes"]
    if _sha256(evaluator_file) != locked_hashes["evaluator_sha256"]:
        raise ValueError("evaluator hash does not match the locked protocol")
    if _sha256(task_root / "task.json") != locked_hashes["task_json_sha256"]:
        raise ValueError("task descriptor hash does not match the locked protocol")
    if _sha256(task_root / "TASK.md") != locked_hashes["task_brief_sha256"]:
        raise ValueError("task brief hash does not match the locked protocol")
    source_files = task["source"]["files"]
    missing = [name for name in source_files if not (workspace / name).is_file()]
    if missing:
        raise FileNotFoundError(f"candidate checkout is missing source files: {missing}")
    revision = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--", *source_files],
        check=True, capture_output=True,
    ).stdout
    acceptance = lock["acceptance"]
    rollout_rows, failures = _grade_rollouts(workspace, acceptance)
    if len(rollout_rows) != acceptance["rollout_conditions"]:
        raise ValueError("fixture rollout count does not match the locked protocol")
    trainer_checks = _trainer_contract(workspace / TRAINER_RELATIVE)
    for check, passed in trainer_checks.items():
        if not passed:
            failures.append({"check": check})
    error_restoration = _grade_error_restoration(workspace, failures)
    invalid_temperatures = _grade_invalid_temperatures(workspace, failures)
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "workspace_revision": revision,
        "candidate_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_file_sha256": {name: _sha256(workspace / name) for name in source_files},
        "evaluator_sha256": _sha256(evaluator_file),
        "task_descriptor_sha256": _sha256(task_root / "task.json"),
        "passed": not failures,
        "checks": {
            "rollout_cases": rollout_rows,
            "trainer": trainer_checks,
            "exception_path": error_restoration,
            "invalid_temperature_cases": invalid_temperatures,
        },
        "failures": failures,
        "resource_scope": {
            "model_weights_loaded": False,
            "third_party_python_packages_used": False,
            "accelerator_hours": 0,
            "paid_api_calls": 0,
            "external_compute_usd": 0,
        },
        "claim_limit": task["claim_limit"],
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-fail", action="store_true",
                        help="return success when the grader ran, even if the candidate failed")
    args = parser.parse_args()
    task_root = args.task_root.resolve()
    try:
        result = grade(args.workspace, task_root)
    except Exception as exc:
        print(json.dumps({"grader_error": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
