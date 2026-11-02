"""Bounded cached-model teacher experiment on newly calibrated Development tasks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(args):
    if digest(args.protocol) != args.protocol_sha256:
        raise ValueError("operator-supplied teacher protocol identity mismatch")
    lock = json.loads(args.protocol.read_text())
    for relative, expected in lock["implementation_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"teacher implementation drift: {relative}")
    for relative, expected in lock["agent_runtime_sha256"].items():
        if digest(args.agent_root / relative) != expected:
            raise ValueError(f"pinned agent dependency drift: {relative}")
    for relative, expected in lock["qualification_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"qualification evidence drift: {relative}")
    if args.monitor is not None:
        for name, expected in lock["monitor_sha256"].items():
            if digest(args.monitor.parent / name) != expected:
                raise ValueError(f"resource monitor drift: {name}")
    return lock


def plain_mistral_messages(flattened):
    """Preserve system/tool content under the checkpoint's user/assistant template."""
    system = flattened[0]["content"]
    result = []
    for index, message in enumerate(flattened[1:]):
        role, content = message["role"], message["content"]
        if index == 0:
            if role != "user":
                raise ValueError("first non-system message must be user")
            content = system + "\n\n" + content
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    return result


def worker(args, lock):
    # Import the attested existing controller and its own dependency tree in a
    # fresh process. The model sees only explicitly allowlisted workspace files.
    import importlib.metadata
    for name, expected in lock["python_packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"inference package version drift: {name}")
    sys.path[:0] = [str(args.agent_root), str(args.agent_root / "src")]
    spec = importlib.util.spec_from_file_location("pinned_agent", args.agent_root / "scripts/run_local_agent.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    from vare.flat_tool_chat import flatten_tool_messages

    trace_path = args.output / "generation-trace.jsonl"

    def record(value):
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def generate(backend, model, tokenizer, messages, tools, *, limit, args):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        rendered = plain_mistral_messages(flatten_tool_messages(messages, tools))
        prompt = tokenizer.apply_chat_template(rendered, chat_template="default", tokenize=False,
                                               add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= args.max_context_tokens:
            raise ValueError("context_budget_exhausted")
        limit = min(limit, args.max_context_tokens-len(ids))
        record({"event": "model_input", "input_text": prompt, "input_ids": ids,
                "max_new_tokens": limit})
        sampler = make_sampler(temp=args.temperature if args.do_sample else 0,
                               top_p=args.top_p if args.do_sample else 0,
                               top_k=args.top_k if args.do_sample else 0)
        segments, token_ids, token_logprobs = [], [], []
        for response in stream_generate(model, tokenizer, prompt=ids, max_tokens=limit, sampler=sampler):
            segments.append(response.text)
            token_ids.append(response.token)
            token_logprobs.append(float(response.logprobs[response.token].item()))
            record({"event": "model_token", "token_id": response.token,
                    "raw_model_logprob": token_logprobs[-1], "text": response.text,
                    "generation_tokens": response.generation_tokens})
        raw = "".join(segments)
        record({"event": "model_output", "text": raw, "token_ids": token_ids,
                "raw_model_token_logprobs": token_logprobs,
                "generated_tokens_including_terminal": response.generation_tokens,
                "finish_reason": response.finish_reason})
        return raw, response.generation_tokens, len(ids)

    agent.generate_turn = generate
    original_public = agent.run_public_tests
    public_count = 0

    def public(*call_args, **call_kwargs):
        nonlocal public_count
        if public_count >= lock["budgets"]["public_test_calls_per_attempt"]:
            return {"passed": False, "error": "public verifier budget exhausted"}
        public_count += 1
        result = original_public(*call_args, **call_kwargs)
        record({"event": "public_test", "call": public_count, "result": result})
        return result

    agent.run_public_tests = public
    attempt = lock["attempts"][args.attempt]
    task = next(task for task in lock["tasks"] if task["id"] == args.task)
    budget = lock["budgets"]
    sys.argv = ["pinned_agent", "--workspace", str(args.output / "workspace"),
                "--task-root", str(args.output / "task"), "--output", str(args.output / "agent.json"),
                "--model-path", str(args.model_path), "--model-repository", lock["model"]["repository"],
                "--backend", "mlx", "--public-test-image", lock["public_test_image"],
                "--instruction", task["instruction"], "--seed", str(attempt["seed"]),
                "--max-turns", str(budget["turns"]), "--max-new-tokens-per-turn", str(budget["tokens_per_turn"]),
                "--max-total-new-tokens", str(budget["tokens_per_attempt"]),
                "--max-context-tokens", str(budget["context_tokens"]),
                "--wall-seconds", str(budget["wall_seconds"]), "--temperature", str(attempt["temperature"]),
                "--top-p", "0.95", "--top-k", "40", "--do-sample" if attempt["sample"] else "--no-do-sample"]
    return agent.main()


def main():
    experiment_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-repository", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--monitor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    lock = verify(args)
    if args.worker:
        return worker(args, lock)

    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    from scripts import qualify_flagship_development_v1 as qualifier
    spec = importlib.util.spec_from_file_location("host_monitor", args.monitor.parent / "monitor_local_agent_resources_v1.py")
    host_monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host_monitor)
    before = host_monitor.observe()
    raw_swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    free_swap_mib = float(raw_swap.split("free = ")[1].split("M")[0])
    preflight = {"observed": before, "raw_swap": raw_swap,
                 "minimum_free_swap_mib": lock["resources"]["minimum_free_swap_mib"],
                 "model_files_verified": False}
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if (before["memory_free_percent"] < lock["resources"]["minimum_free_memory_percent"]
            or free_swap_mib < lock["resources"]["minimum_free_swap_mib"]):
        raise RuntimeError("alternative teacher resource preflight failed; no model loaded")
    for name, expected in lock["model"]["files_sha256"].items():
        if digest(args.model_path / name) != expected:
            raise ValueError(f"model checkpoint mismatch: {name}")
    preflight["model_files_verified"] = True
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    q_protocol = json.loads((ROOT / lock["qualification_protocol"]).read_text())
    summary = {"protocol_sha256": args.protocol_sha256, "attempts": [], "optimizer_steps": 0,
               "training": "NO_GO", "heldout_evaluation": "NOT_RUN", "status": "RUNNING"}
    for task in lock["tasks"]:
        source = qualifier.source_bytes(args.source_repository, task["baseline_commit"], task["target"])
        if hashlib.sha256(source).hexdigest() != task["baseline_sha256"]:
            raise ValueError("upstream baseline source drift")
        if task["id"] == "thinking-markers":
            calls = qualifier.thinking_calls()
        else:
            calls = []
            for case in q_protocol["gptq_cases"]:
                raw = json.loads((ROOT / lock["qualification_run"] / task["id"] / (case["id"]+"-reference.json")).read_text())
                oracle = json.loads(raw["stdout"])
                calls.append(qualifier.grading.FunctionCall(case["id"], "adapter", "run", [oracle["request"]], oracle["expected"]))
        for index, attempt in enumerate(lock["attempts"]):
            directory = args.output / task["id"] / attempt["id"]
            workspace = directory / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "target.py").write_bytes(source)
            public = "import ast\nfrom pathlib import Path\nast.parse(Path('target.py').read_text())\nprint('public syntax check passed')\n"
            (workspace / "test_public.py").write_text(public)
            task_dir = directory / "task"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(json.dumps({
                "id": "flagship-dev-"+task["id"], "prompt": (ROOT / task["prompt"]).read_text(),
                "source": {"upstream_repository": "ml-explore/mlx-lm", "upstream_revision": task["baseline_commit"]},
                "metadata": {"editable_files": ["target.py", "test_public.py"], "allocation": "development"},
            }, indent=2) + "\n")
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"}
            for command in [["git", "init", "-q"], ["git", "add", "target.py", "test_public.py"],
                            ["git", "-c", "user.name=VARE experiment", "-c", "user.email=experiment@invalid", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Pinned baseline export"]]:
                subprocess.run(command, cwd=workspace, env=env, check=True, capture_output=True, timeout=10)
            command = [str(args.python), str(args.monitor), "--output", str(directory / "monitor"),
                       "--max-seconds", str(lock["budgets"]["wall_seconds"]), "--swap-abort-mib",
                       str(before["swap_used_mib"]+lock["resources"]["max_swap_growth_mib"]),
                       "--memory-free-abort-percent", str(lock["resources"]["minimum_free_memory_percent"]),
                       "--rss-abort-mib", str(lock["resources"]["max_process_rss_mib"]), "--",
                       str(args.python), str(Path(__file__).resolve()), "--worker", "--protocol", str(args.protocol.resolve()),
                       "--protocol-sha256", args.protocol_sha256, "--agent-root", str(args.agent_root.resolve()),
                       "--model-path", str(args.model_path.resolve()), "--task", task["id"], "--attempt", str(index),
                       "--output", str(directory.resolve())]
            started = time.monotonic()
            process = subprocess.run(command, capture_output=True, text=True, timeout=lock["budgets"]["wall_seconds"]+30)
            (directory / "launcher.json").write_text(json.dumps({"returncode": process.returncode,
                "stdout": process.stdout, "stderr": process.stderr, "elapsed_seconds": time.monotonic()-started}, indent=2)+"\n")
            monitor = json.loads((directory / "monitor/resource-monitor.json").read_text())
            agent_path = directory / "agent.json"
            if process.returncode or not agent_path.is_file():
                summary["status"] = "ABORTED" if monitor["status"] == "ABORTED_RESOURCE" else "FAILED_INFRASTRUCTURE"
                summary["failure"] = {"task": task["id"], "attempt": attempt["id"], "monitor": monitor["status"]}
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                return 2
            agent_record = json.loads(agent_path.read_text())
            content = (workspace / "target.py").read_bytes()
            if (workspace / "test_public.py").read_text() != public:
                raise ValueError("protected public test changed")
            grading_started = time.monotonic()
            grade = qualifier.grade_source(content, (ROOT / task["adapter"]).read_bytes(), calls, q_protocol)
            grading_elapsed = time.monotonic() - grading_started
            (directory / "independent-grade.json").write_text(json.dumps(asdict(grade), indent=2)+"\n")
            changed = content != source
            success = changed and grade.passed
            outcome = {"task": task["id"], "attempt": attempt["id"], "seed": attempt["seed"],
                       "reward": int(success), "independent_success": success,
                       "source_changed": changed, "new_tokens": agent_record["new_tokens"],
                       "elapsed_seconds": monitor["elapsed_seconds"],
                       "independent_grading_seconds": grading_elapsed,
                       "source_sha256": hashlib.sha256(content).hexdigest(),
                       "model_snapshot": lock["model"]["revision"],
                       "verifier_protocol_sha256": lock["qualification_protocol_sha256"]}
            (directory / "reward.json").write_text(json.dumps(outcome, indent=2)+"\n")
            summary["attempts"].append(outcome)
            print(json.dumps(outcome), flush=True)
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    summary["status"] = "COMPLETED"
    summary["successes"] = sum(item["reward"] for item in summary["attempts"])
    summary["model_run_seconds"] = sum(item["elapsed_seconds"] for item in summary["attempts"])
    summary["total_wall_seconds"] = time.monotonic() - experiment_started
    summary["successes_per_model_run_hour"] = summary["successes"]*3600/summary["model_run_seconds"]
    summary["successes_per_total_wall_hour"] = summary["successes"]*3600/summary["total_wall_seconds"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
