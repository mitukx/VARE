"""``RepoEnv`` with an explicit isolation mode; fails closed by default.

``vare.learning_pipeline.repo_env`` is hash-frozen by
``protocols/coding_policy_learning_mlx_4b_v1.json`` (run-001), and its
``RepoEnv.grade`` executes candidate code with the operator's privileges in a
native virtualenv. Editing that module would invalidate the frozen protocol
(``verify_protocol`` would stop every v1 run with ``protocol_hash_mismatch``),
so the isolation mode lives in this subclass, which later protocols construct
instead of ``env_for``. ``scoring.score_candidate`` and ``scoring.calibrate``
call ``env.grade(...)`` by keyword and accept this class unchanged.

``grade(..., isolation=...)``:

* ``"required"`` (default): grade through the v2 sandboxed grader
  (``learning_pipeline_trusted_verdict.grade_isolated_v2``) using a backend
  whose isolation attestation matches its current configuration. Without one,
  return ``reason: "isolation_unverified"``, ``reward: 0``,
  ``executed: False``; no candidate code runs and the clone is not touched.
* ``"legacy_native"``: explicit opt-in to the frozen v1 path
  (``RepoEnv.grade``), kept so that run-001 can be re-executed as recorded.
  Not a security boundary.

Every result carries an ``isolation`` record: mode, backend, configuration
digest, the attestation used (or why none), the verdict channel and the grader
schema, so rewards from different verifier versions are never pooled silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from scripts import learning_pipeline_isolated_grading as ig
from scripts import learning_pipeline_trusted_verdict as tv
from vare.learning_pipeline.repo_env import RepoEnv

ISOLATION_REQUIRED = "required"
ISOLATION_LEGACY_NATIVE = "legacy_native"
ISOLATION_MODES = (ISOLATION_REQUIRED, ISOLATION_LEGACY_NATIVE)
GRADE_SCHEMA = "learning_pipeline_repo_env_grade_v2"


@dataclass(frozen=True)
class SelectedBackend:
    backend: ig.SandboxBackend | None
    attestation: ig.Attestation | None
    tried: tuple[Mapping[str, Any], ...]


def select_backend(candidates: Sequence[ig.SandboxBackend], *, workspace_root: Path,
                   evidence_dir: Path) -> SelectedBackend:
    """First backend, in the given order, whose isolation probe is fully denied.

    Every attempt is recorded. No candidate passes means no backend (fail closed).
    """
    tried: list[Mapping[str, Any]] = []
    for backend in candidates:
        att = ig.attest(backend, workspace_root=workspace_root, evidence_dir=evidence_dir)
        tried.append({"backend": backend.name, "config_digest": att.config_digest, "ok": att.ok,
                      "failures": list(att.failures)})
        if att.ok:
            return SelectedBackend(backend, att, tuple(tried))
    return SelectedBackend(None, None, tuple(tried))


@dataclass
class TrustedRepoEnv(RepoEnv):
    """A ``RepoEnv`` whose ``grade`` requires attested isolation unless told otherwise.

    ``backend`` must use a dependencies-only interpreter: an editable install of
    the repository would point imports at the host clone instead of the
    sandbox workspace (``sandbox_calibration`` detects this).
    """

    backend: ig.SandboxBackend | None = None
    attestation: ig.Attestation | None = None
    workspace_root: Path | None = None
    import_roots: tuple[str, ...] | None = None
    selection: tuple[Mapping[str, Any], ...] = ()
    static_scan: bool = True

    def isolation_record(self, mode: str) -> dict[str, Any]:
        if mode == ISOLATION_LEGACY_NATIVE:
            return {"mode": mode, "backend": "native_venv", "attested": False, "config_digest": None,
                    "attestation": None, "verdict_channel": "pytest_stdout_v1", "grader": "repo_env_v1",
                    "note": "candidate code runs with operator privileges; not a security boundary"}
        b, att = self.backend, self.attestation
        return {"mode": mode, "backend": b.name if b is not None else None,
                "config_digest": b.config_digest() if b is not None else None,
                "attested": bool(att is not None and att.ok and b is not None
                                 and att.backend == b.name and att.config_digest == b.config_digest()),
                "attestation": att.to_dict() if att is not None else None,
                "selection": [dict(t) for t in self.selection],
                "verdict_channel": tv.VERDICT_CHANNEL, "grader": tv.SCHEMA,
                "runner_sha256": tv.runner_sha256(), "static_scan": self.static_scan}

    def grade(  # type: ignore[override]
        self,
        *,
        bug_patch: str,
        target_file: str,
        candidate_text: str | None,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str],
        timeout: float = 180,
        isolation: str = ISOLATION_REQUIRED,
    ) -> dict[str, object]:
        """Grade one candidate; ``candidate_text=None`` grades the unrepaired bug."""
        if isolation not in ISOLATION_MODES:
            raise ValueError(f"isolation must be one of {ISOLATION_MODES}, got {isolation!r}")
        info = self.isolation_record(isolation)
        if isolation == ISOLATION_LEGACY_NATIVE:
            rec = super().grade(bug_patch=bug_patch, target_file=target_file, candidate_text=candidate_text,
                                fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass, timeout=timeout)
            return {**rec, "schema": GRADE_SCHEMA, "executed": rec.get("reason") != "bug_patch_apply_error",
                    "reward": int(bool(rec.get("passed"))), "isolation": info}
        t0 = time.time()
        f2p_ids, p2p_ids = list(fail_to_pass), list(pass_to_pass)

        def result(passed: bool, reason: str, tests: Mapping[str, str], executed: bool,
                   **extra: Any) -> dict[str, object]:
            f2p = {t: tests.get(t, "NOT_RUN") for t in f2p_ids}
            p2p = {t: tests.get(t, "NOT_RUN") for t in p2p_ids}
            return {"schema": GRADE_SCHEMA, "passed": passed, "reason": reason, "reward": int(passed),
                    "executed": executed,
                    "f2p_passed": sum(v == "PASSED" for v in f2p.values()), "f2p_total": len(f2p),
                    "p2p_passed": sum(v == "PASSED" for v in p2p.values()), "p2p_total": len(p2p),
                    "missing": sorted(t for t, v in {**f2p, **p2p}.items() if v == "MISSING")[:10],
                    "tests": {**f2p, **p2p}, "seconds": round(time.time() - t0, 3),
                    "isolation": info, **extra}

        if not info["attested"] or self.backend is None or self.attestation is None:
            return result(False, "isolation_unverified", {}, False)
        root = self.workspace_root or (self.cache / "grade-workspaces")
        root.mkdir(parents=True, exist_ok=True)
        # Trusted staging only (git checkout/apply, file reads); no candidate code runs here.
        self.reset()
        try:
            try:
                self._apply(bug_patch)
            except RuntimeError as exc:
                return result(False, "bug_patch_apply_error", {}, False, detail=str(exc)[-300:])
            try:
                buggy = self.read(target_file)
            except (OSError, UnicodeDecodeError) as exc:
                return result(False, "target_unreadable", {}, False, detail=str(exc)[-300:])
            v2 = tv.grade_isolated_v2(
                self.backend, self.attestation, workspace_root=root, source_tree=self.path,
                target_file=target_file, buggy_text=buggy,
                candidate_text=buggy if candidate_text is None else candidate_text,
                fail_to_pass=f2p_ids, pass_to_pass=p2p_ids, import_roots=self.import_roots,
                timeout=timeout, static_scan=self.static_scan)
        finally:
            self.reset()
        tests = v2.get("tests") or {}
        keep = {k: v2[k] for k in ("candidate_sha256", "static_flags", "process_exit", "stdout_sha256",
                                   "detail") if k in v2}
        return result(bool(v2.get("passed")), str(v2.get("reason")), tests, bool(v2.get("executed")),
                      **keep)


def trusted_env_for(task: Mapping[str, object], cache: Path, *, backend: ig.SandboxBackend | None,
                    attestation: ig.Attestation | None, **kw: Any) -> TrustedRepoEnv:
    """Counterpart of ``repo_env.env_for`` for protocols after v1."""
    return TrustedRepoEnv(str(task["mirror"]), str(task["upstream_repo"]), str(task["mirror_commit"]), cache,
                          backend=backend, attestation=attestation, **kw)
