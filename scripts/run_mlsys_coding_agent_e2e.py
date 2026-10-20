#!/usr/bin/env python3
"""Resumable MLSys coding-agent runs.

The development protocol is the task-success measurement. Held-out and the
fault comparison refuse to start unless that measurement already contains a
hidden-grader pass. A second process continues from TrajectoryStore when the
episode directory is still marked running.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.coding_agent_loop import (  # noqa: E402
    DurableEpisode,
    WorkspaceTools,
    grade_separated,
    memory_snapshot,
    run_episode,
    sha256_file,
)
from vare.environments.runner import Workspace  # noqa: E402
from vare.environments.spec import EngineeringTaskSpec  # noqa: E402
from vare.trajectory_store import PolicyCheckpoint, TrajectoryStore  # noqa: E402
from vare.types import Task  # noqa: E402


def _reward():
    path = ROOT / "benchmarks" / "mlsys_coding_v1" / "reward.py"
    spec = importlib.util.spec_from_file_location("mlsys_coding_v1_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _code_sha(protocol_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "src" / "vare" / "coding_agent_loop.py",
        ROOT / "src" / "vare" / "local_gguf.py",
        Path(__file__).resolve(),
        protocol_path,
    ):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_rev() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    pack = protocol.get("pack")
    if pack:
        manifest = ROOT / pack.get("root", "benchmarks/mlsys_coding_v1") / "manifest.json"
        if manifest.is_file():
            actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
            expected = pack["manifest_sha256"]
            if actual != expected:
                raise SystemExit(f"pack manifest hash {actual} does not match protocol {expected}")
    return protocol


def _task_dir(task_id: str) -> Path:
    mapping = {
        "mlsys-v1-grad-accum-tail": "grad_accum_tail",
        "mlsys-v1-paired-shuffle-loader": "paired_shuffle_loader",
        "mlsys-v1-range-sums": "range_sums",
        "mlsys-v1-row-log-softmax": "row_log_softmax",
        "mlsys-v1-warmup-cosine": "warmup_cosine",
        "mlsys-v1-aging-scheduler": "aging_scheduler",
        "mlsys-v1-causal-attention": "causal_attention",
        "mlsys-v1-dataset-standardize": "dataset_standardize",
        "mlsys-v1-global-grad-clip": "global_grad_clip",
    }
    directory = ROOT / "benchmarks" / "mlsys_coding_v1" / "tasks" / mapping[task_id]
    if not (directory / "task.json").is_file():
        raise SystemExit(f"missing task directory for {task_id}: {directory}")
    return directory


def _applied(transcript: list[dict]) -> bool:
    for entry in transcript:
        for call in entry.get("tool_calls", []):
            result = call.get("result") or {}
            if call.get("name") in {"replace_text", "write_file", "apply_patch"} and result.get("applied") is True:
                return True
    return False


def _durable(episode: Path, spec: EngineeringTaskSpec, protocol: dict, model_sha: str) -> DurableEpisode:
    store_path = episode / "trajectory.sqlite"
    if store_path.exists():
        store = TrajectoryStore(store_path)
    else:
        store = TrajectoryStore.create(
            store_path,
            run_id="mlsys-coding-agent",
            record_bytes=262144,
            max_inflight=4,
        )
    policy = PolicyCheckpoint("local-gguf", 1, model_sha)
    durable = DurableEpisode(
        store=store,
        trajectory_id=f"task.{spec.id}.s{protocol['seed']}",
        policy=policy,
        worker_id="coder-worker",
        snapshot_dir=episode / "snapshots",
        lease_seconds=float(protocol.get("lease_seconds", 60)),
        heartbeat=True,
    )
    durable.enroll(
        Task(id=spec.id, prompt=spec.prompt[:1500], family=spec.family),
        expected_steps=int(protocol["budgets"]["max_turns"]),
    )
    return durable


def _prepare_workspace(episode: Path, spec: EngineeringTaskSpec, reset: bool) -> Path:
    workspace = episode / "workspace"
    if reset and workspace.exists():
        subprocess.run(["git", "checkout", "--", "."], cwd=workspace, check=True)
        subprocess.run(["git", "clean", "-fd", "-e", ".git"], cwd=workspace, check=True)
    if not (workspace / ".git").exists():
        if workspace.exists():
            raise SystemExit(f"workspace exists without git metadata: {workspace}")
        Workspace(spec, root=workspace)
    return workspace


def run_one(
    *,
    protocol: dict,
    episode: Path,
    task_id: str,
    model_path: Path,
    arm: str,
    fault_after_steps: int | None,
    fault_phase: str | None,
    reset_workspace: bool,
    spent_tokens: int,
    generate=None,
) -> int:
    spec = EngineeringTaskSpec.load(_task_dir(task_id) / "task.json")
    episode.mkdir(parents=True, exist_ok=True)
    workspace = _prepare_workspace(episode, spec, reset_workspace)
    tools = WorkspaceTools(workspace, set(spec.metadata["editable_files"]))
    budgets = protocol["budgets"]
    model = protocol["model"]
    if generate is None:
        from vare.local_gguf import GGUFGenerator

        generate = GGUFGenerator(
            model_path,
            seed=int(protocol["seed"]),
            n_ctx=int(model["n_ctx"]),
            n_threads=int(model["n_threads"]),
            n_batch=int(model.get("n_batch", 128)),
        )
    durable = None
    if arm == "vare":
        durable = _durable(episode, spec, protocol, model["sha256"])
    log_path = episode / "attempts" / f"{time.time_ns()}.jsonl"
    try:
        result = run_episode(
            tools=tools,
            task_prompt=spec.prompt,
            generate=generate,
            max_turns=int(budgets["max_turns"]),
            max_new_tokens=int(budgets["max_new_tokens_per_turn"]),
            max_total_new_tokens=int(budgets["max_total_new_tokens"]),
            max_no_tool_retries=int(budgets["max_no_tool_retries"]),
            wall_seconds=float(budgets["wall_seconds_per_task"]),
            close_think=bool(model.get("close_think", True)),
            durable=durable,
            fault_after_steps=fault_after_steps,
            fault_phase=fault_phase,
            raise_fault=False,
            spent_tokens=spent_tokens,
            attempt_log=log_path,
        )
    except RuntimeError as exc:
        if "lease is still held" in str(exc):
            return 87
        raise
    payload = {
        "task_id": task_id,
        "seed": protocol["seed"],
        "arm": arm,
        "stop_reason": result.stop_reason,
        "new_tokens": result.new_tokens,
        "tool_executions": result.tool_executions,
        "committed_steps": result.committed_steps,
        "elapsed_seconds": result.elapsed_seconds,
        "resumed_from_step": result.resumed_from_step,
        "source_hashes_before": result.source_hashes_before,
        "source_hashes_after": result.source_hashes_after,
        "diff": result.diff,
        "diff_sha256": hashlib.sha256(result.diff.encode()).hexdigest(),
        "transcript": result.transcript,
        "memory": memory_snapshot(),
    }
    (episode / "agent.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _grade(episode: Path, task_id: str) -> dict:
    spec = EngineeringTaskSpec.load(_task_dir(task_id) / "task.json")
    grade = grade_separated(
        task_dir=_task_dir(task_id),
        workspace=episode / "workspace",
        timeout_s=float(spec.limits.wall_time_s),
    )
    decision = _reward().decide(grade)
    agent = json.loads((episode / "agent.json").read_text(encoding="utf-8"))
    verified = bool(decision["success"] and agent["diff"].strip() and _applied(agent["transcript"]))
    decision["verified_task_success"] = verified
    (episode / "grade.json").write_text(json.dumps(grade, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (episode / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def _last_log(episode: Path) -> dict:
    logs = sorted((episode / "attempts").glob("*.jsonl")) if (episode / "attempts").is_dir() else []
    if not logs:
        return {"new_tokens": 0, "tool_executions": 0}
    last = None
    for line in logs[-1].read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = json.loads(line)
    return last or {"new_tokens": 0, "tool_executions": 0}


def _development_passed(summary_path: Path, protocol: dict) -> bool:
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_id") != protocol["requires_development_protocol"]:
        return False
    if summary.get("model_sha256") != protocol["model"]["sha256"]:
        return False
    return any(row.get("verified_task_success") is True for row in summary.get("tasks", []))


def _write_summary(output: Path, protocol: dict, rows: list[dict], code_sha: str) -> None:
    passed = sum(1 for row in rows if row.get("verified_task_success") is True)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "model_sha256": protocol["model"]["sha256"],
        "model_repository": protocol["model"]["repository"],
        "model_revision": protocol["model"]["revision"],
        "code_sha256": code_sha,
        "git_rev": _git_rev(),
        "tasks_completed": len(rows),
        "verified_passes": passed,
        "tasks": rows,
        "memory": memory_snapshot(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def campaign(protocol_path: Path, output: Path, model_path: Path, dev_summary: Path | None) -> int:
    protocol = _load_protocol(protocol_path)
    if protocol["model"]["sha256"] != sha256_file(model_path):
        raise SystemExit("model file sha256 does not match the protocol pin")
    if protocol["kind"] in {"heldout", "recovery"}:
        if dev_summary is None or not _development_passed(dev_summary, protocol):
            raise SystemExit("held-out and recovery stay blocked until a development task passes the hidden grader")
    output.mkdir(parents=True, exist_ok=True)
    code_sha = _code_sha(protocol_path)
    freeze_path = output / "freeze.json"
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
            raise SystemExit("protocol changed after the measurement freeze")
        if freeze["code_sha256"] != code_sha and any(output.glob("**/decision.json")):
            raise SystemExit("runner changed after a graded task; start a new protocol instead of continuing")
    else:
        freeze_path.write_text(json.dumps({
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "code_sha256": code_sha,
            "git_rev": _git_rev(),
            "model_sha256": protocol["model"]["sha256"],
            "memory_before_tasks": memory_snapshot(),
            "created_unix": time.time(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if protocol["kind"] == "recovery":
        return _recovery(protocol, protocol_path, output, model_path, code_sha)
    from vare.local_gguf import GGUFGenerator

    model = protocol["model"]
    print("loading model", flush=True)
    generate = GGUFGenerator(
        model_path,
        seed=int(protocol["seed"]),
        n_ctx=int(model["n_ctx"]),
        n_threads=int(model["n_threads"]),
        n_batch=int(model.get("n_batch", 128)),
    )
    print(json.dumps({"model_loaded": True, "memory": memory_snapshot()}), flush=True)
    rows = []
    for task_id in protocol["tasks"]:
        episode = output / f"seed-{protocol['seed']}" / task_id
        decision_path = episode / "decision.json"
        if decision_path.is_file():
            rows.append(json.loads(decision_path.read_text(encoding="utf-8")) | {"task_id": task_id})
            continue
        print(f"start {task_id}", flush=True)
        rc = 87
        waits = 0
        while rc == 87 and waits < 180:
            rc = run_one(
                protocol=protocol,
                episode=episode,
                task_id=task_id,
                model_path=model_path,
                arm=protocol.get("arm", "vare"),
                fault_after_steps=None,
                fault_phase=None,
                reset_workspace=False,
                spent_tokens=0,
                generate=generate,
            )
            if rc == 87:
                time.sleep(1)
                waits += 1
        if rc != 0:
            raise SystemExit(f"{task_id} worker exited {rc}")
        decision = _grade(episode, task_id)
        decision["task_id"] = task_id
        rows.append(decision)
        _write_summary(output, protocol, rows, code_sha)
        print(json.dumps({"task_id": task_id, "verified_task_success": decision["verified_task_success"], "reason": decision["reason"]}), flush=True)
    _write_summary(output, protocol, rows, code_sha)
    return 0


def _spawn_worker(protocol_path: Path, model_path: Path, episode: Path, task_id: str, arm: str, *, fault: bool, reset: bool, spent: int, phase: str, after_steps: int) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--protocol", str(protocol_path),
        "--model", str(model_path),
        "--episode", str(episode),
        "--task-id", task_id,
        "--arm", arm,
        "--spent-tokens", str(spent),
    ]
    if fault:
        command.extend(["--fault-after-steps", str(after_steps), "--fault-phase", phase])
    if reset:
        command.append("--reset-workspace")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _recovery(protocol: dict, protocol_path: Path, output: Path, model_path: Path, code_sha: str) -> int:
    rows = []
    task_id = protocol["task"]
    for arm in protocol["arms"]:
        episode = output / arm / task_id
        if (episode / "decision.json").is_file():
            rows.append(json.loads((episode / "recovery.json").read_text(encoding="utf-8")))
            continue
        fault_armed = True
        waited = 0.0
        lost_tools = 0
        spent = 0
        started = time.perf_counter()
        while True:
            rc = _spawn_worker(
                protocol_path,
                model_path,
                episode,
                task_id,
                arm,
                fault=fault_armed,
                reset=arm == "baseline" and not fault_armed,
                spent=spent,
                phase=protocol["fault"]["phase"],
                after_steps=int(protocol["fault"]["after_steps"]),
            )
            if rc == 87:
                time.sleep(1)
                waited += 1
                if waited > 180:
                    raise SystemExit("lease was not released")
                continue
            if rc == 86:
                partial = _last_log(episode)
                lost_tools += int(partial.get("tool_executions") or 0)
                spent = int(partial.get("new_tokens") or 0)
                fault_armed = False
                continue
            if rc != 0:
                raise SystemExit(f"recovery worker exited {rc}")
            break
        decision = _grade(episode, task_id)
        record = {
            "arm": arm,
            "task_id": task_id,
            "verified_task_success": decision["verified_task_success"],
            "reason": decision["reason"],
            "lost_tool_executions": lost_tools,
            "restart_wait_seconds": waited,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "new_tokens": json.loads((episode / "agent.json").read_text(encoding="utf-8"))["new_tokens"],
        }
        (episode / "recovery.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(record)
        print(json.dumps(record), flush=True)
    _write_summary(output, protocol, rows, code_sha)
    return 0


def probe(model_path: Path, n_ctx: int, n_threads: int, *, close_think: bool, n_batch: int) -> int:
    from vare.local_gguf import GGUFGenerator
    from vare.coding_agent_loop import parse_tool_calls, render_prompt, system_prompt

    root = Path("/tmp/vare-format-probe")
    if root.exists():
        subprocess.run(["rm", "-rf", str(root)], check=True)
    root.mkdir()
    (root / "sample.py").write_text("VALUE = 0\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "sample.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=VARE", "-c", "user.email=vare@localhost", "commit", "-qm", "baseline"],
        cwd=root,
        check=True,
    )
    tools = WorkspaceTools(root, {"sample.py"}, public_test="test_public.py")
    (root / "test_public.py").write_text("from sample import VALUE\nassert VALUE == 1\n", encoding="utf-8")
    tools.allowed.add("test_public.py")
    prompt = "Set VALUE in sample.py to 1. Use replace_text. Then call finish."
    messages = [
        {"role": "system", "content": system_prompt(prompt)},
        {"role": "user", "content": prompt},
    ]
    generate = GGUFGenerator(model_path, seed=101, n_ctx=n_ctx, n_threads=n_threads, n_batch=n_batch)
    parsed = 0
    for _turn in range(4):
        started = time.perf_counter()
        text = generate(render_prompt(messages, close_think=close_think), 128)
        elapsed = time.perf_counter() - started
        calls = parse_tool_calls(text.text)
        messages.append({"role": "assistant", "content": text.text})
        print(json.dumps({
            "tokens": text.new_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "calls": [call.get("name") or call.get("error") for call in calls],
            "text": text.text[:400],
            "memory": memory_snapshot(),
        }), flush=True)
        if not any("name" in call for call in calls):
            messages.append({"role": "user", "content": "Call replace_text now."})
            continue
        for call in calls:
            if "name" not in call:
                continue
            parsed += 1
            result = tools.execute(call["name"], call["arguments"])
            messages.append({"role": "tool", "content": json.dumps(result)[:2000]})
            if call["name"] == "finish":
                break
        if (root / "sample.py").read_text(encoding="utf-8").strip() == "VALUE = 1":
            record = {"probe_passed": True, "parsed_tool_calls": parsed, "memory": memory_snapshot()}
            print(json.dumps(record), flush=True)
            return 0
    record = {
        "probe_passed": False,
        "parsed_tool_calls": parsed,
        "sample": (root / "sample.py").read_text(encoding="utf-8"),
        "memory": memory_snapshot(),
    }
    print(json.dumps(record), flush=True)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dev-summary", type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--close-think", action="store_true")
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--arm", default="vare")
    parser.add_argument("--fault-after-steps", type=int)
    parser.add_argument("--fault-phase")
    parser.add_argument("--reset-workspace", action="store_true")
    parser.add_argument("--spent-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.probe:
        return probe(args.model, args.n_ctx, args.n_threads, close_think=args.close_think, n_batch=args.n_batch)
    if args.episode is not None:
        protocol = _load_protocol(args.protocol)
        return run_one(
            protocol=protocol,
            episode=args.episode,
            task_id=args.task_id,
            model_path=args.model,
            arm=args.arm,
            fault_after_steps=args.fault_after_steps,
            fault_phase=args.fault_phase,
            reset_workspace=args.reset_workspace,
            spent_tokens=args.spent_tokens,
        )
    if args.protocol is None or args.output is None:
        parser.error("--protocol and --output are required")
    return campaign(args.protocol, args.output, args.model, args.dev_summary)


if __name__ == "__main__":
    raise SystemExit(main())
