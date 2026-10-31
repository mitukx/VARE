#!/usr/bin/env python3
"""Run-002 held-out pool: criterion R7 build check (no model is loaded).

For each candidate repository, in seeded repository order:

  R7a  ``RepoEnv.materialize()`` (clone, HEAD == pinned mirror commit, ``pip install -e .``
       + ``pytest==8.3.5`` + ``EXTRA_TEST_DEPS``; 900 s install timeout) with the interpreter
       running this script (use the run-001 interpreter);
  R7b  venv <= 1.0 GiB and clone <= 300 MB (``du -sk``);
  R7c  for content-filter passers in ascending ``order_key(SEED, id)``: the bug-only grade has
       at least one FAIL_TO_PASS test not PASSED, and the grade with the pinned original file
       restored has every FAIL_TO_PASS and PASS_TO_PASS test PASSED. Failing tasks are skipped
       and the next task in order is tried until ``--tasks-per-repo`` tasks validate.

Every result is written to the JSON report immediately, so an interrupted run resumes where
it stopped (re-run the same command). The report and stdout carry instance IDs, reason codes,
counts, durations, sizes and hashes only: never a problem statement, patch, test name, file
content or absolute host path. Results from a non-macOS host are smoke tests, not R7 evidence
(``environment.r7_host`` is false and the freeze script refuses them by default).

Exit codes: 0 done (or dry run), 2 input/integrity error, 4 stopped on the free-disk guard
(resumable).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run002_heldout_common as common  # noqa: E402
from vare.learning_pipeline import curriculum, edit_format, sources  # noqa: E402
from vare.learning_pipeline.repo_env import GRADER_PYTEST, RepoEnv  # noqa: E402

SCHEMA = "vare.run002.build_check.v1"
INSTALL_TIMEOUT_S = 900  # RepoEnv.materialize's pip timeout (criterion R7a)
VENV_LIMIT_KIB = 1024 * 1024  # 1.0 GiB (R7b)
CLONE_LIMIT_BYTES = 300 * 10**6  # 300 MB (R7b)
DEFAULT_CACHE = Path.home() / ".cache" / "vare-run002-build-check"
DEFAULT_DATASET_CACHE = Path.home() / ".cache" / "vare-learning-pipeline-v1"
DEFAULT_REPORT = common.ROOT / "artifacts" / "learning_pipeline_run002" / "build_check" / "report.json"
PROTOCOL_V1 = common.ROOT / "protocols" / "coding_policy_learning_mlx_4b_v1.json"
FINAL = {"admitted", "insufficient_valid_tasks", "attempt_cap_reached", "env_failed", "size_exceeded",
         "payload_error"}
CODE_FILES = ["scripts/run002_build_check.py", "scripts/run002_heldout_common.py",
              "src/vare/learning_pipeline/repo_env.py", "src/vare/learning_pipeline/curriculum.py",
              "src/vare/learning_pipeline/edit_format.py"]


class InputError(SystemExit):
    """Refusal on inconsistent inputs or report; exit code 2."""

    def __init__(self, msg: str) -> None:
        print(f"error: {msg}", file=sys.stderr)
        super().__init__(2)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def du_kib(path: Path) -> int | None:
    if not path.exists():
        return None
    r = subprocess.run(["du", "-sk", str(path)], capture_output=True, text=True, timeout=600)
    if r.returncode == 0 and r.stdout.split():
        return int(r.stdout.split()[0])
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink()) // 1024


def _tool_version(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout or r.stderr).strip().splitlines()[0] if r.returncode == 0 else None


def environment(python_exe: str) -> dict[str, Any]:
    mac = platform.mac_ver()[0]
    return {
        "system": platform.system(), "machine": platform.machine(), "release": platform.release(),
        "macos_version": mac or None, "r7_host": platform.system() == "Darwin",
        "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
        "venv_python": _tool_version([python_exe, "--version"]),
        "git": _tool_version(["git", "--version"]), "cpu_count": os.cpu_count(),
        "grader_pytest": GRADER_PYTEST,
    }


def sanitize(text: str, cache: Path, dataset_cache: Path) -> str:
    return common.relativize(text, [(cache, "<cache>"), (dataset_cache, "<dataset-cache>")])[-400:]


class Checker:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.cache = Path(a.cache).expanduser().resolve()
        self.dataset_cache = Path(a.dataset_cache).expanduser().resolve()
        self.report_path = Path(a.report).expanduser().resolve()
        self.python = a.python or sys.executable
        self.rows: dict[str, dict[str, Any]] | None = None
        self.selected: list[common.Candidate] = []

    # -------------------------------------------------------------- report
    def load_report(self, inputs: dict[str, str], settings: dict[str, Any]) -> dict[str, Any]:
        if self.report_path.exists():
            rep = json.loads(self.report_path.read_text())
            if rep.get("schema") != SCHEMA or rep.get("inputs") != inputs:
                raise InputError("existing report has a different schema or inputs; use a new --report path")
            if rep.get("settings") != settings:
                raise InputError("existing report used different settings; use a new --report path")
            return rep
        return {"schema": SCHEMA, "criteria_sha256": common.CRITERIA_SHA256, "seed": common.SEED,
                "inputs": inputs, "settings": settings, "started": now(), "runs": [], "repos": {}}

    def save(self, rep: dict[str, Any]) -> None:
        rep["updated"] = now()
        common.atomic_write_json(self.report_path, rep)

    # ------------------------------------------------------------- dataset
    def payloads(self, mirrors: set[str]) -> dict[str, dict[str, Any]]:
        if self.rows is None:
            shards = curriculum.fetch_shards(sources.SWE_SMITH_PY, self.dataset_cache)
            self.rows = {r["instance_id"]: r for r in curriculum.load_rows(shards, mirrors)}
        return self.rows

    # ---------------------------------------------------------------- repo
    def run_repo(self, cand: common.Candidate, rep: dict[str, Any]) -> None:
        a = self.a
        rec = rep["repos"].setdefault(cand.upstream, {
            "group": cand.group, "mirror": cand.mirror, "commit": cand.commit, "license": cand.license,
            "repo_order_key": cand.repo_key, "passers": len(cand.passers), "status": "pending",
            "env": None, "tasks": [], "validated": []})
        if rec["status"] in FINAL:
            return
        env = RepoEnv(cand.mirror, cand.upstream, cand.commit, self.cache)
        if not (rec["env"] or {}).get("ok") or not env.python.exists():
            rec["env"] = self.materialize(env)
            self.save(rep)
            if not rec["env"]["ok"]:
                rec["status"] = rec["env"]["status"]
                self.save(rep)
                return
        else:  # resumed: the env was built and measured in an earlier invocation
            env.reset()
        rec["status"] = "validating"
        tasks = {t["instance_id"]: t for t in rec["tasks"]}
        rows = self.payloads({c.mirror for c in self.selected})
        cfg = json.loads((common.V1_CONFIG_DIR / "curriculum.json").read_text())
        for iid in cand.passers:
            if len(rec["validated"]) >= a.tasks_per_repo:
                break
            if iid in tasks:
                continue
            if a.max_attempts_per_repo and len(rec["tasks"]) >= a.max_attempts_per_repo:
                rec["status"] = "attempt_cap_reached"
                break
            res = self.validate(env, iid, rows.get(iid), cand, cfg)
            rec["tasks"].append(res)
            if res["valid"]:
                rec["validated"].append(iid)
            self.save(rep)
        if rec["status"] == "validating":
            rec["status"] = ("admitted" if len(rec["validated"]) >= a.tasks_per_repo
                             else "insufficient_valid_tasks")
        env.reset()
        self.save(rep)
        if a.prune:
            shutil.rmtree(env.venv, ignore_errors=True)
            shutil.rmtree(env.path, ignore_errors=True)

    def materialize(self, env: RepoEnv) -> dict[str, Any]:
        # A venv without a recorded successful build may be a torn install: rebuild it so the
        # recorded time and size describe one complete, unattended build (R7a).
        shutil.rmtree(env.venv, ignore_errors=True)
        t0 = time.time()
        out: dict[str, Any] = {"ok": False, "started": now()}
        try:
            env.materialize(python_exe=self.python)
        except subprocess.TimeoutExpired as exc:
            step = "install" if "pip" in " ".join(map(str, exc.cmd)) else "git"
            out.update(status="env_failed", reason=f"{step}_timeout")
        except RuntimeError as exc:
            msg = str(exc)
            reason = next((code for key, code in (("clone failed", "clone_failed"), ("HEAD", "head_mismatch"),
                                                  ("venv failed", "venv_failed"),
                                                  ("install failed", "install_failed")) if key in msg),
                          "materialize_error")
            out.update(status="env_failed", reason=reason, error_tail=sanitize(msg, self.cache, self.dataset_cache))
        else:
            out.update(ok=True, status="built", reason="ok")
        out["seconds"] = round(time.time() - t0, 1)
        out["venv_kib"] = du_kib(env.venv)
        out["clone_kib"] = du_kib(env.path)
        if out["ok"]:
            if out["venv_kib"] is not None and out["venv_kib"] > VENV_LIMIT_KIB:
                out.update(ok=False, status="size_exceeded", reason="venv_too_large")
            elif out["clone_kib"] is not None and out["clone_kib"] * 1024 > CLONE_LIMIT_BYTES:
                out.update(ok=False, status="size_exceeded", reason="clone_too_large")
        return out

    def validate(self, env: RepoEnv, iid: str, row: dict[str, Any] | None, cand: common.Candidate,
                 cfg: dict[str, Any]) -> dict[str, Any]:
        res: dict[str, Any] = {"instance_id": iid, "order_key": common.task_order_key(iid), "valid": False}
        t0 = time.time()
        if row is None or row["repo"] != cand.mirror:
            res["reason"] = "missing_from_pinned_dataset"
            return res
        f2p, p2p = list(row["FAIL_TO_PASS"] or []), list(row["PASS_TO_PASS"] or [])
        res["hashes"] = {"patch_sha256": curriculum.sha256_text(row["patch"]),
                         "problem_statement_sha256": curriculum.sha256_text(row["problem_statement"]),
                         "fail_to_pass_sha256": common.sha256_json(f2p),
                         "pass_to_pass_sha256": common.sha256_json(p2p)}
        res["fail_to_pass_count"], res["pass_to_pass_count"] = len(f2p), len(p2p)
        try:
            target = edit_format.parse_single_file_patch(row["patch"]).path
        except ValueError:
            res["reason"] = "unsupported_patch"
            return res
        if not all("::" in t for t in f2p + p2p):
            # The v1 grader selects tests by pytest node id; a unittest-style id can never be
            # reported PASSED by RepoEnv.grade, so the reference grade cannot pass (R7c fails).
            # Short-circuit with the same outcome instead of running pytest twice.
            res["reason"] = "non_pytest_test_ids"
            return res
        task = {"_patch": row["patch"], "target_file": target}
        why = curriculum._file_check(env, task, cfg)  # builder content filter, re-run on this host
        if why:
            res["reason"] = why
            return res
        timeout = self.a.grade_timeout
        base = env.grade(bug_patch=row["patch"], target_file=target, candidate_text=None,
                         fail_to_pass=f2p, pass_to_pass=p2p, timeout=timeout)
        res["baseline"] = summarize(base)
        res["baseline_p2p_green"] = base.get("p2p_passed") == base.get("p2p_total") and "p2p_total" in base
        if base["reason"] == "bug_patch_apply_error":
            res["reason"] = "baseline_patch_apply_error"
        elif base["reason"] == "test_timeout":
            res["reason"] = "baseline_timeout"
        elif base.get("f2p_passed", 0) >= base.get("f2p_total", 0):
            res["reason"] = "baseline_does_not_fail"
        if "reason" in res:
            res["seconds"] = round(time.time() - t0, 2)
            return res
        env.reset()
        original = env.read(target)
        ref = env.grade(bug_patch=row["patch"], target_file=target, candidate_text=original,
                        fail_to_pass=f2p, pass_to_pass=p2p, timeout=timeout)
        res["reference"] = summarize(ref)
        if ref["passed"]:
            res.update(valid=True, reason="valid")
        elif ref["reason"] == "test_timeout":
            res["reason"] = "reference_timeout"
        elif ref["reason"] == "protected_file_changed":
            res["reason"] = "reference_protected_file_changed"
        elif ref.get("missing"):
            res["reason"] = "reference_tests_missing"
        else:
            res["reason"] = "reference_fails"
        res["seconds"] = round(time.time() - t0, 2)
        return res

    # ----------------------------------------------------------------- run
    def run(self) -> int:
        a = self.a
        inputs = common.verify_inputs()
        cands = common.load_candidates(min_passers=a.min_passers)
        if a.repos:
            want = set(a.repos.split(","))
            unknown = want - {c.upstream for c in cands}
            if unknown:
                print(f"unknown or ineligible repos: {sorted(unknown)}", file=sys.stderr)
                return 2
            cands = [c for c in cands if c.upstream in want]
        if a.limit:
            cands = cands[: a.limit]
        self.selected = cands
        settings = {"tasks_per_repo": a.tasks_per_repo, "min_passers": a.min_passers,
                    "grade_timeout_s": a.grade_timeout, "install_timeout_s": INSTALL_TIMEOUT_S,
                    "venv_limit_kib": VENV_LIMIT_KIB, "clone_limit_bytes": CLONE_LIMIT_BYTES,
                    "max_attempts_per_repo": a.max_attempts_per_repo}
        if a.dry_run:
            return self.dry_run(cands, settings)
        rep = self.load_report(inputs, settings)
        rep["runs"].append({"started": now(), "environment": environment(self.python),
                            "code_sha256": {f: common.sha256_file(common.ROOT / f) for f in CODE_FILES},
                            "repos_requested": [c.upstream for c in cands]})
        rep["environment"] = rep["runs"][-1]["environment"]
        self.save(rep)
        for c in cands:
            free = shutil.disk_usage(self.cache if self.cache.exists() else self.cache.parent).free / 2**30
            if free < a.min_free_gib:
                print(f"stop: free disk {free:.1f} GiB < {a.min_free_gib} GiB (resume after freeing space)")
                self.save(rep)
                return 4
            t0 = time.time()
            self.run_repo(c, rep)
            r = rep["repos"][c.upstream]
            env = r.get("env") or {}
            print(f"{c.upstream:34s} {r['status']:24s} env={env.get('reason')} {env.get('seconds')}s "
                  f"venv={env.get('venv_kib')}KiB tried={len(r['tasks'])} valid={len(r['validated'])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rep["runs"][-1]["finished"] = now()
        self.save(rep)
        done = [r for r in rep["repos"].values() if r["status"] == "admitted"]
        print(f"report: {common.rel_label(self.report_path)}; repos admitted with "
              f"{a.tasks_per_repo} tasks: {len(done)}/{len(rep['repos'])}")
        return 0

    def dry_run(self, cands: list[common.Candidate], settings: dict[str, Any]) -> int:
        shard_dir = self.dataset_cache / "datasets" / sources.SWE_SMITH_PY.source_id / sources.SWE_SMITH_PY.revision
        have = sum((shard_dir / Path(s).name).exists() for s in sources.SWE_SMITH_PY.shards)
        print(f"dry run: inputs verified (criteria {common.CRITERIA_SHA256[:8]}), {len(cands)} repos, "
              f"settings {json.dumps(settings, sort_keys=True)}")
        print(f"dataset shards cached: {have}/{len(sources.SWE_SMITH_PY.shards)}; interpreter "
              f"{platform.python_version()} on {platform.system()} (R7 host: {platform.system() == 'Darwin'})")
        prior = json.loads(self.report_path.read_text())["repos"] if self.report_path.exists() else {}
        for i, c in enumerate(cands, 1):
            env = RepoEnv(c.mirror, c.upstream, c.commit, self.cache)
            st = prior.get(c.upstream, {}).get("status", "-")
            print(f"{i:2d} {c.upstream:34s} grp={c.group} {c.license:12s} passers={len(c.passers):3d} "
                  f"clone={'y' if env.path.exists() else 'n'} venv={'y' if env.python.exists() else 'n'} "
                  f"report={st}")
        return 0


def summarize(g: dict[str, Any]) -> dict[str, Any]:
    """Grade summary without test names, stdout or patch-apply detail."""
    keys = ("passed", "reason", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total", "pytest_exit",
            "stdout_sha256", "seconds", "pytest_seconds")
    out = {k: g[k] for k in keys if k in g}
    out["missing_count"] = len(g.get("missing") or [])
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="clones and venvs (separate from run-001)")
    ap.add_argument("--dataset-cache", default=str(DEFAULT_DATASET_CACHE),
                    help="cache holding the pinned SWE-smith-py shards (run-001 cache; downloaded if absent)")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--python", default=None, help="interpreter for the venvs (default: this interpreter)")
    ap.add_argument("--tasks-per-repo", type=int, default=common.MAX_TASKS_PER_REPO)
    ap.add_argument("--min-passers", type=int, default=common.MAX_TASKS_PER_REPO,
                    help="candidate repos need at least this many passers (default 8: the 35 R1-R6 repos)")
    ap.add_argument("--max-attempts-per-repo", type=int, default=0, help="0 = no cap (criterion R7c)")
    ap.add_argument("--grade-timeout", type=float,
                    default=float(json.loads(PROTOCOL_V1.read_text())["budgets"]["grading"]["timeout_s"]))
    ap.add_argument("--repos", default="", help="comma-separated subset (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="first N repos in seeded order (smoke tests)")
    ap.add_argument("--min-free-gib", type=float, default=4.0)
    ap.add_argument("--prune", action="store_true", help="delete clone and venv after each repo")
    ap.add_argument("--dry-run", action="store_true", help="verify inputs and print the plan; no network")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return Checker(a).run()
    except RuntimeError as exc:
        print(f"input error: {common.relativize(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
