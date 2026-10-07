from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .demo import run_demo
from .config import LagConfig
from .controls import RVLOuterPlanner
from .evidence import CapabilityScore, artifact_manifest
from .integrations.rvl import RVLTokenReplayReader
from .protocol import ProtocolLock
from .environments import CommandWorkspaceAgent, EngineeringTaskSpec, EnvironmentCampaignRunner, ExecutableEvaluator, TaskCatalog, Workspace
from .evidence import sha256_json


def _write_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_demo(args: argparse.Namespace) -> int:
    results = asyncio.run(run_demo(rounds=args.rounds, rollouts=args.rollouts, seed=args.seed))
    rows = [asdict(r) for r in results]
    if args.output:
        _write_json(args.output, rows)
    for r in results:
        status = "PROMOTE" if r.decision.accepted else "REJECT"
        lcb = "" if r.decision.paired_lcb is None else f" paired_lcb={r.decision.paired_lcb:+.3f}"
        print(
            f"round={r.round_index} {status} incumbent={r.incumbent_eval.primary:.3f} "
            f"candidate={r.candidate_eval.primary:.3f} gain={r.decision.primary_gain:+.3f}{lcb} active={r.promoted_id}"
        )
    return 0


def _cmd_rvl_inspect(args: argparse.Namespace) -> int:
    reader = RVLTokenReplayReader(args.replay_sqlite)
    snapshot = reader.snapshot(
        current_policy_version=args.current_policy_version,
        current_verifier_version=args.current_verifier_version,
    )
    payload = snapshot.to_dict()
    if args.include_ready:
        experiences = reader.ready_experiences(
            current_policy_version=args.current_policy_version,
            current_verifier_version=args.current_verifier_version,
            trusted_default=args.trusted_default,
        )
        payload["ready_experiences"] = len(experiences)
        payload["ready_failures"] = sum(not e.verification.passed for e in experiences)
        payload["ready_mean_policy_lag"] = (
            sum(e.policy_lag for e in experiences) / len(experiences) if experiences else 0.0
        )
        payload["ready_mean_verifier_lag"] = (
            sum(e.verifier_lag for e in experiences) / len(experiences) if experiences else 0.0
        )
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0




def _cmd_rvl_plan(args: argparse.Namespace) -> int:
    reader = RVLTokenReplayReader(args.replay_sqlite)
    planner = RVLOuterPlanner(
        LagConfig(
            max_policy_lag=args.max_policy_lag,
            max_verifier_lag=args.max_verifier_lag,
        )
    )
    plan = planner.plan(
        reader,
        current_policy_version=args.current_policy_version,
        current_verifier_version=args.current_verifier_version,
        trusted_default=args.trusted_default,
    )
    payload = plan.to_dict()
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

def _cmd_lock_protocol(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    lock = ProtocolLock.create(
        name=str(raw["name"]),
        config=dict(raw.get("config", {})),
        hypotheses=tuple(raw.get("hypotheses", [])),
        primary_metrics=tuple(raw.get("primary_metrics", [])),
        acceptance=dict(raw.get("acceptance", {})),
    )
    lock.write(args.output)
    print(json.dumps(lock.payload(), indent=2, sort_keys=True))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    score = CapabilityScore(
        before=args.before,
        after=args.after,
        gpu_hours=args.gpu_hours,
        rollout_tokens=args.rollout_tokens,
        learner_tokens=args.learner_tokens,
        verifier_cost=args.verifier_cost,
    )
    payload = score.to_dict()
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    payload = [asdict(x) for x in artifact_manifest(args.paths)]
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0



def _baseline_metric_map(payload: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in payload.get("metrics", []):
        try:
            out[str(row["name"])] = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _cmd_env_validate(args: argparse.Namespace) -> int:
    spec = EngineeringTaskSpec.load(args.spec)
    payload = {
        "task_id": spec.id,
        "family": spec.family,
        "tests": len(spec.tests),
        "metrics": len(spec.metrics),
        "protected_paths": len(spec.protected_paths),
        "spec_sha256": sha256_json(spec.to_dict()),
    }
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_env_verify(args: argparse.Namespace) -> int:
    spec = EngineeringTaskSpec.load(args.spec)
    baseline = None
    if args.baseline_report:
        baseline = _baseline_metric_map(json.loads(Path(args.baseline_report).read_text(encoding="utf-8")))
    result = ExecutableEvaluator(spec, evaluator_root=Path(args.spec).resolve().parent).evaluate(
        args.workspace,
        baseline_metrics=baseline,
        provenance={"mode": "env-verify"},
    )
    payload = result.to_dict()
    if args.output:
        _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.passed else 2


def _cmd_env_smoke(args: argparse.Namespace) -> int:
    spec = EngineeringTaskSpec.load(args.spec)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    evaluator = ExecutableEvaluator(spec, evaluator_root=Path(args.spec).resolve().parent)
    with Workspace(spec) as workspace:
        baseline = evaluator.evaluate(workspace.path, provenance={"phase": "baseline"})
        _write_json(outdir / "baseline.json", baseline.to_dict())
        if not args.oracle_patch:
            payload = {"baseline": baseline.to_dict(), "oracle": None}
            _write_json(outdir / "summary.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            return 0
        import subprocess
        proc = subprocess.run(
            ["python", args.oracle_patch, str(workspace.path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"oracle patch failed: {proc.stderr}")
        oracle = evaluator.evaluate(
            workspace.path,
            baseline_metrics=_baseline_metric_map(baseline.to_dict()),
            provenance={"phase": "oracle-harness-check"},
        )
        _write_json(outdir / "oracle.json", oracle.to_dict())
        payload = {
            "baseline_passed": baseline.passed,
            "baseline_score": baseline.score,
            "oracle_passed": oracle.passed,
            "oracle_score": oracle.score,
            "spec_sha256": sha256_json(spec.to_dict()),
        }
        _write_json(outdir / "summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0 if (not baseline.passed and oracle.passed) else 3


def _cmd_env_campaign(args: argparse.Namespace) -> int:
    catalog = TaskCatalog.discover(args.catalog_root)
    argv = json.loads(args.agent_argv_json)
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("--agent-argv-json must encode a non-empty JSON string array")
    agent = CommandWorkspaceAgent(tuple(argv), timeout_s=args.agent_timeout_s)
    runner = EnvironmentCampaignRunner(
        catalog,
        agent,
        run_root=args.run_root,
        verifier_version=args.verifier_version,
    )
    summary = asyncio.run(runner.run(repeats=args.repeats, task_ids=args.task_id or None))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0

def main() -> int:
    p = argparse.ArgumentParser(prog="vare", description="Verification-Aware Reinforcement Engine")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the deterministic contextual-bandit capability loop")
    d.add_argument("--rounds", type=int, default=8)
    d.add_argument("--rollouts", type=int, default=512)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--output", type=str)
    d.set_defaults(func=_cmd_demo)

    r = sub.add_parser("rvl-inspect", help="inspect an RVL TokenReplay SQLite database read-only")
    r.add_argument("--replay-sqlite", required=True)
    r.add_argument("--current-policy-version", type=int, required=True)
    r.add_argument("--current-verifier-version", type=int, required=True)
    r.add_argument("--include-ready", action="store_true")
    r.add_argument("--trusted-default", action="store_true")
    r.add_argument("--output")
    r.set_defaults(func=_cmd_rvl_inspect)

    rp = sub.add_parser("rvl-plan", help="derive a safe outer-loop curriculum/control plan from RVL replay")
    rp.add_argument("--replay-sqlite", required=True)
    rp.add_argument("--current-policy-version", type=int, required=True)
    rp.add_argument("--current-verifier-version", type=int, required=True)
    rp.add_argument("--max-policy-lag", type=int, default=2)
    rp.add_argument("--max-verifier-lag", type=int, default=2)
    rp.add_argument("--trusted-default", action="store_true")
    rp.add_argument("--output")
    rp.set_defaults(func=_cmd_rvl_plan)

    l = sub.add_parser("lock-protocol", help="freeze a preregistered experiment spec with a SHA-256 digest")
    l.add_argument("--spec", required=True)
    l.add_argument("--output", required=True)
    l.set_defaults(func=_cmd_lock_protocol)

    s = sub.add_parser("score", help="compute capability gain per GPU-hour")
    s.add_argument("--before", type=float, required=True)
    s.add_argument("--after", type=float, required=True)
    s.add_argument("--gpu-hours", type=float, required=True)
    s.add_argument("--rollout-tokens", type=int, default=0)
    s.add_argument("--learner-tokens", type=int, default=0)
    s.add_argument("--verifier-cost", type=float, default=0.0)
    s.add_argument("--output")
    s.set_defaults(func=_cmd_score)

    m = sub.add_parser("manifest", help="hash evidence artifacts for immutable provenance")
    m.add_argument("paths", nargs="+")
    m.add_argument("--output")
    m.set_defaults(func=_cmd_manifest)

    ev = sub.add_parser("env-validate", help="validate and hash an executable repository task spec")
    ev.add_argument("--spec", required=True)
    ev.add_argument("--output")
    ev.set_defaults(func=_cmd_env_validate)

    e = sub.add_parser("env-verify", help="run trusted executable evaluation against an existing workspace")
    e.add_argument("--spec", required=True)
    e.add_argument("--workspace", required=True)
    e.add_argument("--baseline-report")
    e.add_argument("--output")
    e.set_defaults(func=_cmd_env_verify)

    es = sub.add_parser("env-smoke", help="exercise the repository-environment harness on a baseline and optional oracle patch")
    es.add_argument("--spec", default="benchmarks/smoke/stable_logsumexp/task.json")
    es.add_argument("--oracle-patch", default="tests/fixtures/oracle/fix_stable_logsumexp.py")
    es.add_argument("--output-dir", default="artifacts/env-smoke")
    es.set_defaults(func=_cmd_env_smoke)

    ec = sub.add_parser("env-campaign", help="run a fixed-budget repository-agent campaign over a task catalog")
    ec.add_argument("--catalog-root", required=True)
    ec.add_argument("--agent-argv-json", required=True, help="JSON array; supports {workspace}, {prompt}, {step} placeholders")
    ec.add_argument("--agent-timeout-s", type=float, default=1800.0)
    ec.add_argument("--repeats", type=int, default=1)
    ec.add_argument("--task-id", action="append")
    ec.add_argument("--verifier-version", type=int, default=0)
    ec.add_argument("--run-root", required=True)
    ec.set_defaults(func=_cmd_env_campaign)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
