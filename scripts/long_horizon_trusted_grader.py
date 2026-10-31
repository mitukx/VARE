"""Trusted grader factory for ``scripts/long_horizon_eval.py`` (v2 isolated grader).

Builds, per Train task, a ``TrustedRepoEnv`` (``scripts/learning_pipeline_trusted_repo_env.py``)
whose sandbox backend passed its isolation attestation, then admits the task only if, inside
that sandbox, the reference file passes and the unrepaired bug fails. Every other case fails
closed: the returned grader reports a reason and ``long_horizon_eval.trusted_verdict`` never
trusts it. The v2 grader modules are imported lazily, so the harness itself does not depend
on them.

Backends: ``docker`` (one dependencies-only image per repository, pinned by digest) for the
Mac; ``bwrap`` (dependencies-only host venv) for Linux hosts and tests.
"""
from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from vare.evidence import sha256_json

ADMISSION_SCHEMA = "long_horizon_grader_admission_v1"


class RefusedGrader:
    """Grader stand-in for a task that is not admitted; its results are never trusted."""

    version = 2

    def __init__(self, reason: str, detail: Mapping[str, Any]):
        self.reason = reason
        self.detail = dict(detail)
        self.name = f"lp_grade_v2_refused.{sha256_json(self.detail)[:16]}"

    def config_sha256(self) -> str:
        return sha256_json({"refused": self.reason, **self.detail})

    def grade(self, *, task: Any, files: Mapping[str, str]) -> Mapping[str, Any]:
        return {"schema": None, "executed": False, "passed": False, "reason": self.reason}


class TrustedV2Grader:
    """Grades the single target file through ``TrustedRepoEnv.grade(isolation="required")``."""

    version = 2

    def __init__(self, env: Any, record: Mapping[str, Any], *, timeout: float):
        self.env = env
        self.record = record
        self.timeout = timeout
        self.name = f"lp_grade_v2.{self.config_sha256()[:16]}"

    def config_sha256(self) -> str:
        info = self.env.isolation_record("required")
        return sha256_json({k: info.get(k) for k in ("mode", "backend", "config_digest", "verdict_channel",
                                                      "grader", "runner_sha256", "static_scan")}
                           | {"import_roots": list(self.env.import_roots or ())})

    def _grade(self, candidate: str | None) -> dict[str, Any]:
        r = self.record
        return dict(self.env.grade(bug_patch=r["_patch"], target_file=r["target_file"], candidate_text=candidate,
                                   fail_to_pass=r["fail_to_pass"], pass_to_pass=r["pass_to_pass"],
                                   timeout=self.timeout, isolation="required"))

    def grade(self, *, task: Any, files: Mapping[str, str]) -> Mapping[str, Any]:
        target = self.record["target_file"]
        if set(files) != {target}:
            return {"schema": None, "executed": False, "passed": False, "reason": "grades_single_target_file"}
        return self._grade(files[target])


def _reference_text(env: Any, target: str) -> str:
    return subprocess.run(["git", "show", f"{env.commit}:{target}"], cwd=env.path, check=True,
                          capture_output=True, text=True).stdout


def make_factory(*, backend: str, cache: Path, evidence_dir: Path, images: Mapping[str, str] | None = None,
                 venv: Path | None = None, timeout: float = 180) -> Callable[[Mapping[str, Any], Any], Any]:
    """Return ``factory(record, env) -> grader`` for ``long_horizon_eval.run``."""
    # Optional dependency (PR #170); a missing module raises here, before any episode runs.
    ig: Any = importlib.import_module("scripts.learning_pipeline_isolated_grading")
    tre: Any = importlib.import_module("scripts.learning_pipeline_trusted_repo_env")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = cache / "lh-grade-workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)

    def sandbox(record: Mapping[str, Any]) -> Any:
        src_layout = str(record["target_file"]).startswith("src/")
        if backend == "docker":
            image = (images or {}).get(str(record["upstream_repo"]))
            if not image or "@sha256:" not in image:
                return None
            roots = ("/workspace/src",) if src_layout else ("/workspace",)
            return ig.DockerBackend(image=image, workspace_root=workspace_root, import_roots=roots)
        if backend == "bwrap":
            if venv is None:
                return None
            return ig.BwrapBackend(venv=venv, import_roots=("/work/src",) if src_layout else ("/work",))
        raise ValueError(f"unknown backend {backend!r}")

    def factory(record: Mapping[str, Any], env: Any) -> Any:
        iid = str(record["instance_id"])
        b = sandbox(record)
        log: dict[str, Any] = {"schema": ADMISSION_SCHEMA, "instance_id": iid, "backend": backend}
        if b is None:
            grader: Any = RefusedGrader("isolation_unverified_no_backend", {"instance_id": iid, "backend": backend})
        else:
            sel = tre.select_backend([b], workspace_root=workspace_root, evidence_dir=evidence_dir / "probe")
            log["selection"] = [dict(t) for t in sel.tried]
            trusted_env = tre.trusted_env_for(record, cache, backend=sel.backend, attestation=sel.attestation,
                                              selection=sel.tried, workspace_root=workspace_root,
                                              import_roots=b.import_roots)
            grader = TrustedV2Grader(trusted_env, record, timeout=timeout)
            if sel.backend is None:
                grader = RefusedGrader("isolation_unverified", {"instance_id": iid, "tried": log["selection"]})
            else:
                ref = grader._grade(_reference_text(env, str(record["target_file"])))
                bug = grader._grade(None)
                log["reference"] = {k: ref.get(k) for k in ("passed", "reason", "executed", "f2p_passed",
                                                            "f2p_total", "p2p_passed", "p2p_total")}
                log["bug"] = {k: bug.get(k) for k in ("passed", "reason", "executed", "f2p_passed", "f2p_total")}
                admitted = ref.get("passed") is True and bug.get("executed") is True and bug.get("passed") is False
                log["admitted"] = admitted
                if not admitted:
                    grader = RefusedGrader("task_not_admitted", {"instance_id": iid,
                                                                 "reference": log["reference"],
                                                                 "bug": log["bug"]})
        log["grader"] = grader.name
        (evidence_dir / f"{iid}.json").write_text(json.dumps(log, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8")
        return grader

    return factory
