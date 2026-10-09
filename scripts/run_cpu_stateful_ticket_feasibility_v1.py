#!/usr/bin/env python3
"""Frozen base-only feasibility screen for a generated ticket-tool task."""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import os
import platform
import random
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PROTOCOL = ROOT / "protocols/cpu_qwen_ticket_tool_feasibility_v1.lock.json"
SEED = 7182026
N_PER_FAMILY = 12
FAMILIES = ("assign_then_progress", "verify_then_resolve", "resolve_then_archive", "two_assign_then_progress")
SYSTEM = "You operate a ticket-management tool. Return only a JSON array of tool calls. Do not add prose or markdown. Use the exact object form {\"tool\":name,...} and only the listed fields. Tools: assign_ticket(ticket, assignee); add_tag(ticket, tag); set_status(ticket, status); archive_ticket(ticket). assignee is Ari, Bo, Caro, Dee, or Eli. tag is billing, bug, login, mobile, shipping, or verified. status is in_progress or resolved. A ticket can enter in_progress only after it has an assignee. A ticket can be resolved only after it has the verified tag. A ticket can be archived only after it is resolved. Calls in the array execute from first to last."
MAX_WALL_SECONDS = 1200
MAX_RSS_BYTES = 22 * 1024**3
MAX_NEW_TOKENS = 128


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def make_ticket(rng: random.Random, ticket_id: str) -> dict[str, Any]:
    return {
        "ticket": ticket_id,
        "status": rng.choice(("open", "in_progress")),
        "assignee": rng.choice(("Ari", "Bo", "Caro", "Dee", "Eli")),
        "priority": rng.choice(("low", "normal", "high")),
        "tags": sorted(rng.sample(("billing", "bug", "login", "mobile", "shipping"), 2)),
        "archived": False,
    }


def call(tool: str, **fields: str) -> dict[str, str]:
    return {"tool": tool, **fields}


def task_prompt(family: str, initial: list[dict[str, Any]], request: str) -> tuple[str, str]:
    user = (
        "Current ticket state (JSON):\n"
        + json.dumps(initial, sort_keys=True, separators=(",", ":"))
        + "\n\nRequest:\n"
        + request
        + "\n\nReturn the minimal ordered JSON tool-call array needed to carry out the request."
    )
    return SYSTEM, user


def build_tasks() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    tasks: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for offset in range(N_PER_FAMILY):
            ix = family_index * N_PER_FAMILY + offset
            target_id = f"TK-{2000 + ix:04d}"
            initial = [make_ticket(rng, target_id)]
            while len(initial) < 5:
                distractor_id = f"TK-{9000 + len(initial) * 37 + ix:04d}"
                initial.append(make_ticket(rng, distractor_id))
            if family == "assign_then_progress":
                initial[0]["status"] = "open"
                initial[0]["assignee"] = ""
            elif family == "two_assign_then_progress":
                initial[0]["status"] = "open"
                initial[0]["assignee"] = ""
                initial[1]["status"] = "open"
                initial[1]["assignee"] = ""
            elif family == "resolve_then_archive":
                initial[0]["tags"] = sorted(set(initial[0]["tags"]) | {"verified"})
            before = copy.deepcopy(initial)
            target = initial[0]
            precedence_pairs: list[list[dict[str, str]]] = []
            if family == "assign_then_progress":
                assignee = rng.choice(("Ari", "Bo", "Caro", "Dee", "Eli"))
                request = f"Assign {target_id} to {assignee}, then move it to in_progress."
                calls = [call("assign_ticket", ticket=target_id, assignee=assignee), call("set_status", ticket=target_id, status="in_progress")]
                precedence_pairs = [[calls[0], calls[1]]]
                target["assignee"], target["status"] = assignee, "in_progress"
            elif family == "verify_then_resolve":
                request = f"Add the verified tag to {target_id}, then resolve it."
                calls = [call("add_tag", ticket=target_id, tag="verified"), call("set_status", ticket=target_id, status="resolved")]
                precedence_pairs = [[calls[0], calls[1]]]
                target["tags"] = sorted((*target["tags"], "verified"))
                target["status"] = "resolved"
            elif family == "resolve_then_archive":
                request = f"Resolve {target_id} and then archive it. Keep every other ticket unchanged."
                calls = [call("set_status", ticket=target_id, status="resolved"), call("archive_ticket", ticket=target_id)]
                precedence_pairs = [[calls[0], calls[1]]]
                target["status"], target["archived"] = "resolved", True
            else:
                second = initial[1]
                assignees = ["Ari", "Bo", "Caro", "Dee", "Eli"]
                rng.shuffle(assignees)
                assignee_a, assignee_b = assignees[:2]
                request = f"Assign {target_id} to {assignee_a}, then move it to in_progress. Assign {second['ticket']} to {assignee_b}, then move it to in_progress. Leave all other tickets unchanged."
                calls = [
                    call("assign_ticket", ticket=target_id, assignee=assignee_a),
                    call("set_status", ticket=target_id, status="in_progress"),
                    call("assign_ticket", ticket=second["ticket"], assignee=assignee_b),
                    call("set_status", ticket=second["ticket"], status="in_progress"),
                ]
                precedence_pairs = [[calls[0], calls[1]], [calls[2], calls[3]]]
                target["assignee"], target["status"] = assignee_a, "in_progress"
                second["assignee"], second["status"] = assignee_b, "in_progress"
            initial.sort(key=lambda row: row["ticket"])
            before.sort(key=lambda row: row["ticket"])
            expected = copy.deepcopy(initial)
            assert len({row["ticket"] for row in initial}) == 5
            system, user = task_prompt(family, before, request)
            tasks.append({
                "task_id": f"{family_index:02d}-{offset:02d}",
                "family": family,
                "system_prompt": system,
                "user_prompt": user,
                "request": request,
                "initial_state": before,
                "expected_state": expected,
                "authorized_calls": calls,
                "precedence_pairs": precedence_pairs,
            })
    return tasks


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if sys.version.split()[0] != protocol["runtime"]["python"] or platform.machine() != "arm64" or sys.platform != "darwin":
        raise RuntimeError("runtime does not match the frozen macOS arm64/Python protocol")
    if protocol.get("runner_sha256") != file_digest(Path(__file__)):
        raise RuntimeError("runner hash differs from the frozen protocol")
    auditor_path = ROOT / "scripts/audit_cpu_stateful_ticket_feasibility_v1.py"
    if protocol.get("auditor_sha256") != file_digest(auditor_path):
        raise RuntimeError("auditor hash differs from the frozen protocol")
    if protocol.get("task", {}).get("seed") != SEED or protocol.get("task", {}).get("families") != {
        "assign_then_progress": "Assign an unassigned ticket, then move it to in_progress; the status transition requires an assignee.",
        "verify_then_resolve": "Add a missing verified tag, then resolve the ticket; resolution requires the verified tag.",
        "resolve_then_archive": "Resolve one ticket, then archive it; archiving before resolution is invalid.",
        "two_assign_then_progress": "For each of two tickets, assign it before moving it to in_progress; the two per-ticket chains must each preserve their own order.",
    }:
        raise RuntimeError("task generator constants differ from the frozen protocol")
    if protocol["runtime"]["max_wall_seconds"] != MAX_WALL_SECONDS or protocol["runtime"]["max_peak_rss_bytes"] != MAX_RSS_BYTES or protocol["runtime"]["max_new_tokens"] != MAX_NEW_TOKENS or protocol["runtime"]["threads"] != 4:
        raise RuntimeError("runtime constants differ from the frozen protocol")
    (out / "protocol.lock.json").write_bytes(protocol_bytes)
    tasks = build_tasks()
    task_pack_path = out / "task-pack.json"
    task_pack_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    results_path = out / "responses.json"
    results_path.write_text("[]\n", encoding="utf-8")
    metadata_path = out / "run-metadata.json"
    metadata = {
        "status": "in_progress",
        "protocol_sha256": digest(protocol_bytes),
        "protocol_id": protocol["protocol_id"],
        "runner_sha256": file_digest(Path(__file__)),
        "auditor_sha256": file_digest(auditor_path),
        "task_pack_sha256": file_digest(task_pack_path),
        "response_sha256": file_digest(results_path),
        "model_revision": protocol["model"]["revision"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "sys_platform": sys.platform,
        },
        "device": "cpu",
        "precision": "fp32",
        "network": "offline",
        "paid_compute": False,
        "threads": 4,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "task_count": len(tasks),
        "completed_responses": 0,
        "started_unix_time": time.time(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    def persist_uncaught(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        metadata.update({"status": "failed", "finished_unix_time": time.time(), "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes(), "error_type": exc_type.__name__, "error": str(exc)})
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sys.__excepthook__(exc_type, exc, tb)

    def persist_interrupted() -> None:
        if metadata.get("status") == "in_progress":
            metadata.update({"status": "interrupted_before_completion", "finished_unix_time": time.time(), "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes()})
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sys.excepthook = persist_uncaught
    atexit.register(persist_interrupted)
    def wall_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("wall-time hard limit reached")

    signal.signal(signal.SIGALRM, wall_timeout)
    signal.alarm(MAX_WALL_SECONDS)
    model_path = args.model_path.resolve(strict=True)
    if not model_path.as_posix().endswith(protocol["model"]["snapshot_path_suffix"]):
        raise RuntimeError("model path does not match the frozen snapshot path")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    if torch.__version__ != protocol["runtime"]["torch"] or transformers.__version__ != protocol["runtime"]["transformers"]:
        raise RuntimeError("torch or transformers version differs from the frozen protocol")
    metadata["runtime"].update({"torch": torch.__version__, "transformers": transformers.__version__})
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable for this CPU-only protocol")
    model_files = {}
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            model_files[path.name] = digest(path.read_bytes())
    if model_files != protocol["model"]["files_sha256"]:
        raise RuntimeError("cached model files differ from the frozen hashes")
    metadata["model_files_sha256"] = model_files
    metadata["model_path"] = str(model_path)
    metadata["task_pack_sha256"] = file_digest(task_pack_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        model.to("cpu")
        model.eval()
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("model parameters are not on CPU")
        load_rss = peak_rss_bytes()
        if load_rss > MAX_RSS_BYTES:
            raise MemoryError("peak RSS gate exceeded after model loading")
        results = []
        for task in tasks:
            messages = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_prompt"]},
            ]
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if encoded.ndim != 2 or encoded.shape[0] != 1:
                raise RuntimeError("unexpected chat-template tensor shape")
            input_ids = encoded.to("cpu")
            with torch.inference_mode():
                output = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            if output.device.type != "cpu":
                raise RuntimeError("generated output tensor is not on CPU")
            response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            results.append({"task_id": task["task_id"], "response": response, "prompt_sha256": digest(task["user_prompt"].encode()), "output_tokens": int(output.shape[1] - input_ids.shape[1])})
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            elapsed = time.monotonic() - started
            rss = peak_rss_bytes()
            metadata.update({"completed_responses": len(results), "response_sha256": file_digest(results_path), "elapsed_seconds": elapsed, "peak_rss_bytes": rss})
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if elapsed > MAX_WALL_SECONDS:
                raise TimeoutError("wall-time gate exceeded; partial response list retained")
            if rss > MAX_RSS_BYTES:
                raise MemoryError("peak RSS gate exceeded; partial response list retained")
        metadata.update({"status": "inference_complete_ungraded", "finished_unix_time": time.time(), "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes(), "model_path": str(model_path), "model_files_sha256": model_files})
        signal.alarm(0)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except BaseException as exc:
        metadata.update({"status": "failed", "finished_unix_time": time.time(), "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": peak_rss_bytes(), "error_type": type(exc).__name__, "error": str(exc)})
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
