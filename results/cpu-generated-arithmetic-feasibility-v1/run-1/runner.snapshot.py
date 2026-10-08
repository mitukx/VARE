#!/usr/bin/env python3
"""Run the frozen train-only generated arithmetic rollout feasibility check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_generated_arithmetic_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_generated_arithmetic_feasibility_v1.lock.json"
DATA_PATH = ROOT / "data/cpu-arithmetic-feasibility-v1/pilot.json"
GENERATOR_PATH = ROOT / "scripts/generate_compositional_arithmetic_feasibility_pilot.py"
MODEL_DIR = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
             "7ae557604adf67be50417f59c2c2f167def9a775")
OUTPUT = ROOT / "results/cpu-generated-arithmetic-feasibility-v1/run-1"
ANSWER_RE = re.compile(r"\s*FINAL:\s*(-?\d+)\s*", re.IGNORECASE)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_spec() -> tuple[dict[str, Any], str]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = dict(lock)
    locked_hash = locked.pop("sha256", None)
    observed = hashlib.sha256(canonical(spec)).hexdigest()
    if locked_hash != observed or canonical(locked) != canonical(spec):
        raise ValueError("feasibility protocol differs from its frozen lock")
    return spec, observed


def rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


class ResourceGuard:
    def __init__(self, seconds: int, memory: int):
        self.seconds, self.memory = seconds, memory
        self.stop = threading.Event()

    def __enter__(self):
        def abort(signum, _frame):
            raise RuntimeError("resource guard aborted pilot with signal %s" % signum)
        self.old_usr1 = signal.signal(signal.SIGUSR1, abort)
        self.old_alarm = signal.signal(signal.SIGALRM, abort)
        signal.alarm(self.seconds)
        def monitor():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.memory:
                    os.kill(os.getpid(), signal.SIGUSR1)
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.monitor_thread.join(timeout=1.0)
        signal.alarm(0)
        signal.signal(signal.SIGUSR1, self.old_usr1)
        signal.signal(signal.SIGALRM, self.old_alarm)
        return False


def write_manifest(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "audit.json"):
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    write_json(root / "manifest.json", {"algorithm": "sha256", "files": files})


def parse_answer(text: str):
    match = ANSWER_RE.fullmatch(text)
    return int(match.group(1)) if match else None


def run(output: Path) -> dict[str, Any]:
    spec, protocol_hash = load_spec()
    started = time.monotonic()
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite pilot output: " + str(output))
    output.mkdir(parents=True)
    for name, source in (("protocol.json", SPEC_PATH), ("protocol.lock.json", LOCK_PATH),
                         ("pilot.json", DATA_PATH)):
        (output / name).write_bytes(source.read_bytes())
    (output / "generator.snapshot.py").write_bytes(GENERATOR_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    auditor = ROOT / "scripts/audit_cpu_generated_arithmetic_feasibility_v1.py"
    (output / "auditor.snapshot.py").write_bytes(auditor.read_bytes())

    try:
        if sha256_file(DATA_PATH) != spec["data"]["pilot_rows_sha256"]:
            raise ValueError("frozen pilot data hash differs")
        if sha256_file(GENERATOR_PATH) != spec["data"]["generator_sha256"]:
            raise ValueError("frozen data generator hash differs")
        model_hashes = {}
        for name, expected in spec["model"]["files_sha256"].items():
            path = MODEL_DIR / name
            if not path.is_file() or sha256_file(path) != expected:
                raise FileNotFoundError("pinned cached model asset unavailable or changed: " + name)
            model_hashes[name] = expected

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.__version__.split("+")[0] != spec["runtime"]["torch"]:
            raise RuntimeError("PyTorch version differs from the frozen runtime")
        if transformers.__version__ != spec["runtime"]["transformers"]:
            raise RuntimeError("Transformers version differs from the frozen runtime")
        if sys.version_info[:3] != tuple(int(x) for x in spec["runtime"]["python"].split(".")):
            raise RuntimeError("Python version differs from the frozen runtime")
        torch.set_num_threads(spec["compute_limits"]["threads"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized; CPU-only run refused")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32,
            use_safetensors=True, low_cpu_mem_usage=True)
        model.to("cpu")
        model.eval()
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise RuntimeError("model parameters are not all on CPU")

        rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        if len(rows) != spec["data"]["pilot_rows"]:
            raise ValueError("frozen pilot row count differs")
        records = []
        with ResourceGuard(spec["compute_limits"]["max_wall_seconds"],
                           spec["compute_limits"]["max_peak_rss_bytes"]):
            for row in rows:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                direct_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt")
                if not torch.equal(input_ids["input_ids"], direct_ids):
                    raise ValueError("chat template string/token path mismatch")
                input_count = int(input_ids["input_ids"].shape[1])
                if input_count > spec["compute_limits"]["max_input_tokens"]:
                    raise ValueError("pilot prompt exceeds frozen input token limit")
                started_row = time.monotonic()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids=input_ids["input_ids"], attention_mask=input_ids["attention_mask"],
                        do_sample=False, max_new_tokens=spec["compute_limits"]["max_new_tokens"],
                        repetition_penalty=spec["generation"]["repetition_penalty"],
                        eos_token_id=spec["generation"]["eos_token_ids"],
                        pad_token_id=spec["generation"]["pad_token_id"],
                        use_cache=True)
                continuation = output_ids[0, input_count:]
                generated_text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
                parsed = parse_answer(generated_text)
                elapsed_row = time.monotonic() - started_row
                records.append({
                    "pilot_index": row["pilot_index"], "composition": row["composition"],
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                    "expression": row["expression"], "oracle_answer": row["oracle_answer"],
                    "generated_text": generated_text, "parsed_answer": parsed,
                    "parsed": parsed is not None,
                    "exact_match": parsed == row["oracle_answer"],
                    "input_tokens": input_count, "generated_tokens": int(continuation.numel()),
                    "wall_seconds": elapsed_row,
                })
                if time.monotonic() - started > spec["compute_limits"]["max_wall_seconds"]:
                    raise TimeoutError("pilot exceeded frozen wall-time limit")
                if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                    raise MemoryError("pilot exceeded frozen peak-RSS limit")

        exact_count = sum(bool(row["exact_match"]) for row in records)
        parsed_count = sum(bool(row["parsed"]) for row in records)
        parsed_wrong = sum(row["parsed"] and not row["exact_match"] for row in records)
        limits_respected = (rss_bytes() <= spec["compute_limits"]["max_peak_rss_bytes"] and
                            time.monotonic() - started <= spec["compute_limits"]["max_wall_seconds"])
        decision_spec = spec["decision"]["pass_conditions"]
        passed = (len(records) == spec["data"]["pilot_rows"] and
                  decision_spec["exact_match_count_minimum"] <= exact_count <= decision_spec["exact_match_count_maximum"] and
                  parsed_count / len(records) >= decision_spec["parse_rate_minimum"] and
                  parsed_wrong >= decision_spec["parsed_wrong_answer_count_minimum"] and limits_respected)
        summary = {
            "protocol_id": spec["protocol_id"], "protocol_canonical_sha256": protocol_hash,
            "phase": spec["phase"], "pilot_rows_sha256": sha256_file(DATA_PATH),
            "generator_sha256": sha256_file(GENERATOR_PATH),
            "runner_sha256": sha256_file(Path(__file__)), "auditor_sha256": sha256_file(auditor),
            "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "model": {"id": spec["model"]["id"], "revision": spec["model"]["revision"],
                      "files_sha256": model_hashes},
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
            "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
            "mps_used": False,
            "network_disabled": True, "paid_compute": False, "threads": torch.get_num_threads(),
            "rows": records,
            "metrics": {"pilot_rows": len(records), "exact_match_count": exact_count,
                        "exact_match_rate": exact_count / len(records), "parsed_count": parsed_count,
                        "parse_rate": parsed_count / len(records), "parsed_wrong_answer_count": parsed_wrong,
                        "unparsed_count": len(records) - parsed_count},
            "decision": {"pass": bool(passed), "compute_limits_respected": bool(limits_respected),
                         "changes_next_action": "design separate formal development protocol" if passed else "stop this task setup"},
            "peak_rss_bytes": rss_bytes(), "total_wall_seconds": time.monotonic() - started,
            "formal_update_performed": False,
        }
        write_json(output / "summary.json", summary)
        write_json(output / "environment.json", {"runtime": summary["runtime"],
                   "model_file_sha256": model_hashes, "device": "cpu", "network_disabled": True,
                   "paid_compute": False})
        write_manifest(output)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__,
                   "message": str(exc), "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        summary = run(args.output)
    except Exception as exc:
        print("pilot failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "output": str(args.output.resolve()),
                      "decision": summary["decision"], "metrics": summary["metrics"],
                      "wall_seconds": summary["total_wall_seconds"],
                      "peak_rss_bytes": summary["peak_rss_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
