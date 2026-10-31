"""Measure durable synthetic agent execution, verification and replay eligibility.

No neural model, optimizer, semantic task evaluation or promotion is simulated
as real learning. Sync batches and continuous async use the same worker counts,
durability, generated work and signed admission checks. Output must be new.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vare.durable import FenceError
from vare.evidence import sha256_file, sha256_json
from vare.trajectory_admission import TrajectoryRewardRoot, admit_terminal_grpo_group
from vare.trajectory_store import AgentStep, BackpressureError, PolicyCheckpoint, TrajectoryStore, audit_trajectory_store
from vare.trajectory_workers import execute_trajectories, owned_io, verify_trajectories
from vare.types import Task, Verification


POLICY = PolicyCheckpoint("synthetic-policy-0", 0, sha256_json({"synthetic_policy": "deterministic-counter-v1", "source_sha256": sha256_file(__file__)}))
VERIFIER = sha256_json({"synthetic_verifier": "sum-of-step-indices-v1"})
RULE = "terminal_action_on_policy_grpo_v1"


def workload(seed, group):
    rng = random.Random(f"{seed}:group:{group}")
    return min(64, 2 + int(rng.paretovariate(1.4) * 2))


def build_step(index, state, total, ident, delay=0.):
    value = state.get("sum", 0) + index + 1
    last = index == total - 1
    action = {"kind": "answer"} if last else {"kind": "tool", "operation_id": f"{ident}:tool:{index}", "semantics": "at_least_once"}
    action["synthetic_delay_s"] = delay
    return AgentStep(index, f"step-{index}", POLICY, action, {"value": index + 1}, {"sum": value},
                     {"prompt_id": f"task-{int(ident.split('-')[0][1:]):06d}",
                      "prompt": f"Return the final counter {value}", "response": str(value),
                      "logprob": 0., "token_count": 1, "latency_s": 0.,
                      "metadata": {"backend": "synthetic-counter", "model": "deterministic-counter-v1",
                                   "sampling_temperature": 1., "logprob_distribution": "neutral_temperature_scaled_policy",
                                   "prompt_token_ids": [0, value],
                                   "response_token_ids": [value], "response_token_logprobs": [0.]}})


def killed_worker(path, pipe):
    store = TrajectoryStore(path)
    # The claim deadline is computed before the claim transaction's synchronous
    # (fsync'd) commit, so a 0.1 s claim lease can expire before append_step on a
    # runner with slow durable commits (reproduced by delaying fsync in this
    # process). Claim with a generous lease so the first step is always persisted,
    # then shorten the deadline to 0.1 s with a heartbeat so it still expires
    # promptly after the parent's SIGKILL; the recovery path is unchanged.
    lease = store.claim("execute", "sigkill-worker", lease_seconds=5)
    resumed = store.resume_state(lease.trajectory_id)
    store.append_step(lease, build_step(0, {}, resumed["spec"]["expected_steps"], lease.trajectory_id))
    store.heartbeat(lease, lease_seconds=.1)
    pipe.send(asdict(lease))
    pipe.recv()


def quantiles(values):
    values = sorted(values)
    return {f"p{n}": values[min(len(values) - 1, math.ceil(len(values) * n / 100) - 1)] for n in (50, 95, 99)}


async def campaign(args, output, protocol):
    run_id = f"trajectory-{args.seed}-{args.mode}"
    store = TrajectoryStore.create(output / "live.db", run_id=run_id,
                                   max_inflight=protocol["max_inflight"], storage_bytes=protocol["storage_bytes"])
    store.register_policy(POLICY)
    store.register_verifier("synthetic-oracle", 0, VERIFIER)
    root = TrajectoryRewardRoot(os.urandom(32), run_id, sha256_json(protocol))
    # This local systems fixture keeps the signing key in the coordinator, not
    # in the killed agent process. It is not adversarial process isolation.
    stop = asyncio.Event()
    observations = {"backpressure_retries": 0, "max_backlog": 0, "peak_storage_bytes": 0,
                    "duplicate_ack_rejections": 0, "completed": 0, "consumer_idle_s": 0.}
    stats = [{} for _ in range(args.workers + 2)]
    latency, verification_latency = [], []
    workers = []
    started = time.perf_counter()
    cpu_start = time.process_time()
    recovery = {"sigkill": False}
    raw = (output / "latencies.jsonl").open("x")

    async def enroll(group, member):
        ident = f"g{group:06d}-{member}"
        task = Task(f"task-{group:06d}", "Execute bounded counter tools", family="synthetic-counter",
                    metadata={"learning_rule": RULE, "seed": args.seed})
        while True:
            try:
                await owned_io(store.submit, ident, task, expected_steps=workload(args.seed, group),
                               group_id=f"g{group:06d}", member=member, group_size=2, lifetime_seconds=1200)
                return
            except BackpressureError:
                observations["backpressure_retries"] += 1
                await asyncio.sleep(.005)

    async def step_fn(resumed, ident):
        index = resumed["next_step"]
        rng = random.Random(f"{args.seed}:{ident}:{index}")
        delay = min(.02, .0005 * rng.paretovariate(1.5))
        await asyncio.sleep(delay)
        return build_step(index, resumed["state"], resumed["spec"]["expected_steps"], ident, delay)

    async def verify_fn(store, ident):
        began = time.perf_counter()
        row = await owned_io(store.trajectory, ident)
        n = row["expected_steps"]
        # A separate verifier consumes saved observations rather than trusting
        # the agent's final reward/state field. No whole trace is loaded.
        def grade():
            total, count = 0, 0
            for record in store.iter_steps(ident):
                count += 1
                total += record["observation"]["value"]
            return count == n and total == n * (n + 1) // 2
        correct = await owned_io(grade)
        elapsed = time.perf_counter() - began
        value = Verification(float(correct), correct, 1., 0, "synthetic-oracle", trusted=True,
                             metadata={"trajectory_sha256": row["head"], "verifier_config_sha256": VERIFIER,
                                       "verification_wall_s": elapsed, "kind": "synthetic-known-ground-truth"})
        return root.attest(ident, row["head"], value)

    async def producer():
        for first in range(0, args.trajectories // 2, max(1, args.workers // 2)):
            for group in range(first, min(args.trajectories // 2, first + max(1, args.workers // 2))):
                for member in range(2):
                    await enroll(group, member)
            if args.mode == "sync":
                expected = min(args.trajectories, (first + max(1, args.workers // 2)) * 2)
                while observations["completed"] < expected:
                    await asyncio.sleep(.002)

    async def consumer():
        last_progress = time.perf_counter()
        while observations["completed"] < args.trajectories:
            progress = False
            for ident in await owned_io(store.verified_groups, 32):
                result = await owned_io(admit_terminal_grpo_group, store, ident, active_policy=POLICY,
                                        verifier_name="synthetic-oracle", verifier_version=0,
                                        verifier_config_sha256=VERIFIER, trust_root=root)
                if not result.admitted:
                    if result.reasons not in {("incomplete_group",), ("unverified_group",)}:
                        raise RuntimeError(f"unexpected admission failure: {result.reasons}")
                    continue
                rows = await owned_io(store.group, ident)
                await owned_io(store.close_group, ident, result.heads,
                               decision={"rule": RULE, "kind": "eligibility-only-no-optimizer"})
                ended = time.time()
                for row in rows:
                    elapsed = ended - row["created"]
                    v = json.loads(row["verification"])["metadata"]["verification_wall_s"]
                    latency.append(elapsed)
                    verification_latency.append(v)
                    raw.write(json.dumps({"trajectory_id": row["id"], "e2e_s": elapsed, "verification_s": v,
                                          "steps": row["next_step"], "trace_sha256": row["head"]}) + "\n")
                observations["completed"] += len(rows)
                progress = True
                if int(ident[1:]) % 128 == 0:
                    try:
                        await owned_io(store.close_group, ident, result.heads)
                    except FenceError:
                        observations["duplicate_ack_rejections"] += 1
                    else:
                        raise RuntimeError("duplicate effective acknowledgment accepted")
            if not progress:
                began = time.perf_counter()
                await asyncio.sleep(.002)
                observations["consumer_idle_s"] += time.perf_counter() - began
            elif time.perf_counter() - last_progress > .2:
                last_progress = time.perf_counter()
            for worker in workers:
                if worker.done():
                    worker.result()
                    raise RuntimeError("worker exited before campaign completion")

    async def monitor():
        while not stop.is_set():
            counts = await owned_io(store.counts)
            observations["max_backlog"] = max(observations["max_backlog"], sum(v for k, v in counts.items() if k not in {"consumed", "quarantined", "cancelled"}))
            size = 0
            for suffix in ("", "-wal", "-shm"):
                try:
                    size += Path(str(store.path) + suffix).stat().st_size
                except FileNotFoundError:
                    pass
            observations["peak_storage_bytes"] = max(observations["peak_storage_bytes"], size)
            if recovery.get("sigkill") and "next_step_s" not in recovery:
                row = await owned_io(store.trajectory, "g000000-0")
                if row["next_step"] >= 2:
                    recovery["next_step_s"] = time.perf_counter() - recovery["killed_at_perf"]
            await asyncio.sleep(.02)

    await enroll(0, 0)
    if args.sigkill:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=killed_worker, args=(store.path, child))
        process.start()
        try:
            if not await asyncio.to_thread(parent.poll, 15):
                raise RuntimeError("fault worker did not persist first step")
            old_lease = parent.recv()
            os.kill(process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.join, 10)
            if process.exitcode != -signal.SIGKILL:
                raise RuntimeError("worker was not SIGKILLed")
            recovery.update(sigkill=True, exitcode=process.exitcode, killed_at_perf=time.perf_counter(), old_lease=old_lease)
        finally:
            if process.is_alive():
                process.kill()
                process.join(5)
            parent.close()
            child.close()
    workers.extend(asyncio.create_task(execute_trajectories(store, f"worker-{i}", step_fn, stop, stats=stats[i])) for i in range(args.workers))
    workers.extend(asyncio.create_task(verify_trajectories(store, f"verifier-{i}", verify_fn, stop, stats=stats[args.workers + i])) for i in range(2))
    sampler = asyncio.create_task(monitor())
    try:
        async with asyncio.timeout(1200):
            await asyncio.gather(producer(), consumer())
    finally:
        stop.set()
        for task in [*workers, sampler]:
            task.cancel()
        await asyncio.gather(*workers, sampler, return_exceptions=True)
        raw.close()
    elapsed = time.perf_counter() - started
    cpu = time.process_time() - cpu_start
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if recovery["sigkill"]:
        from vare.trajectory_store import TrajectoryLease
        try:
            store.append_step(TrajectoryLease(**recovery["old_lease"]), build_step(1, {"sum": 1}, workload(args.seed, 0), "g000000-0"))
        except FenceError:
            recovery["stale_worker_rejected"] = True
        else:
            raise RuntimeError("stale worker commit accepted")
        recovery.pop("killed_at_perf")
    snapshot_hash = await owned_io(store.backup, output / "snapshot.db")
    audit = await owned_io(audit_trajectory_store, output / "snapshot.db")
    if audit["completions"] != args.trajectories or store.counts() != {"consumed": args.trajectories}:
        raise RuntimeError("campaign coverage mismatch")
    expected_steps = 2 * sum(workload(args.seed, group) for group in range(args.trajectories // 2))
    if audit["steps"] != expected_steps:
        raise RuntimeError("effective steps differ from declared workload")
    return {"pass": True, "kind": "synthetic-control-plane-eligibility-only", "mode": args.mode,
            "seed": args.seed, "workers": args.workers, "verifiers": 2, "trajectories": args.trajectories,
            "steps": expected_steps, "wall_s": elapsed, "trajectories_per_s": args.trajectories / elapsed,
            "steps_per_s": expected_steps / elapsed, "timed_events": audit["events"] - 2,
            "events_per_s": (audit["events"] - 2) / elapsed,
            "e2e_latency_s": quantiles(latency), "verification_latency_s": quantiles(verification_latency),
            "parent_peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "parent_cpu_core_percent": 100 * cpu / elapsed, "observations": observations,
            "worker_stats": stats, "recovery": recovery, "audit": audit,
            "snapshot_sha256": snapshot_hash, "latency_samples": len(latency),
            "real_optimizer_updates": 0, "promotion_correctness": "not exercised",
            "scope": "Real wall time, SQLite persistence and OS worker kill; synthetic inference/tool workload. Parent RSS excludes killed child. Consumer idle means polling sleep; worker idle excludes DB claim time. No optimizer or promotion claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=75011)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=("sync", "async"), default="async")
    parser.add_argument("--sigkill", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if not 2 <= args.trajectories <= 50000 or args.trajectories % 2 or not 1 <= args.workers <= 64:
            raise ValueError("invalid bounded workload")
        repository = Path(__file__).resolve().parents[1]
        protocol = json.loads((repository / "benchmarks/trajectory_runtime_v2/protocol.json").read_text())
        if (args.seed not in protocol["seeds"] or args.workers != protocol["workers"]
                or args.mode not in protocol["modes"] or not args.sigkill
                or args.trajectories not in {protocol["stage_a_trajectories"], protocol["stage_b_trajectories"], protocol["optional_stage_c_trajectories"]}):
            raise ValueError("condition is outside the fixed trajectory-runtime v2 protocol")
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
        sources = {str(path.relative_to(repository)): sha256_file(path) for path in [Path(__file__), *[repository / "src/vare" / name for name in ("trajectory_store.py", "trajectory_workers.py", "trajectory_admission.py", "durable.py", "runner.py", "evidence.py", "types.py")]]}
        for name in sources:
            target = args.output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, target)
        provenance = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
                      "source_files": sources, "python": sys.version, "platform": platform.platform(),
                      "condition": vars(args) | {"output": str(args.output)}, "protocol_sha256": sha256_json(protocol)}
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        result = asyncio.run(campaign(args, args.output, protocol))
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        hashes = {str(p.relative_to(args.output)): sha256_file(p) for p in args.output.rglob("*") if p.is_file() and p.name != "live.db" and not p.name.startswith("live.db-")}
        (args.output / "SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except BaseException as exc:
        (args.output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}) + "\n")
        raise


if __name__ == "__main__":
    main()
