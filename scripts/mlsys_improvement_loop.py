"""Propose -> implement -> independently verify loop for one MLSys improvement.

A proposer (scripted, an LLM, or a person) submits a unified diff against the
rollout edit-apply module (``src/vare/learning_pipeline/edit_format.py``). The
parent then decides ACCEPT or REJECT under the pre-registered rule in
``protocols/mlsys_improvement_loop_v1.json``. Stages, in order, each fail closed:

1. ``admission``  - the diff may only modify ``edit_format.py`` in place (no
   new, deleted, renamed, mode-changed or binary entries) and must change it;
2. ``policy``     - AST allowlist for the whole candidate module: imports
   from a small standard-library set only; no dynamic code, file, attribute
   or interpreter access (time, os, sys, open, getattr, exec, dunders, ...);
3. ``static``     - the v2 grader's blocking patterns on added lines;
4. ``tests``      - public + held-out tests through the v2 trusted runner in
   the attested bubblewrap sandbox (HMAC frames judged by the parent; skipped
   or missing tests fail);
5. ``differential`` - outputs on fresh seeded edge-heavy inputs must equal
   the unmodified module's outputs computed by the parent;
6. ``timing``     - interleaved, randomized baseline/candidate/empty-control
   sandbox runs with a parent-side clock, per-run fresh seeds, outputs checked
   every run, an A/A noise-floor calibration, and a paired Student-t interval
   on log speedups.

Usage (repository root, ``PYTHONPATH=src``)::

    python -m scripts.mlsys_improvement_loop export-view --out VIEW_DIR
    python -m scripts.mlsys_improvement_loop evaluate --patch P.diff --source llm \
        --description "..." --out OUT_DIR
    python -m scripts.mlsys_improvement_loop suite --out results/mlsys-improvement-loop-v1
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
from typing import Any, Mapping, Sequence

from scripts import learning_pipeline_isolated_grading as ig
from scripts import learning_pipeline_trusted_verdict as tv
from vare.isolated_candidate import paired_difference_ci, workload_covers_startup

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_improvement_loop_v1"
PROTOCOL_PATH = ROOT / "protocols" / "mlsys_improvement_loop_v1.json"
TARGET_REL = "src/vare/learning_pipeline/edit_format.py"
MODULE = "edit_format"
MODULE_FILE = MODULE + ".py"
WORKLOAD_REL = "benchmarks/mlsys_improvement_loop_v1/workload.py"
PUBLIC_TESTS_REL = "benchmarks/mlsys_improvement_loop_v1/proposer_view/test_public_edit_format.py"
TASK_REL = "benchmarks/mlsys_improvement_loop_v1/proposer_view/task.md"
HIDDEN_TESTS_REL = "benchmarks/mlsys_improvement_loop_v1/hidden/test_hidden_edit_format.py"
BENCH_RUNNER_REL = "scripts/mlsys_improvement_bench_runner.py"
TRUSTED_RUNNER_REL = "scripts/learning_pipeline_trusted_runner.py"
PROPOSALS_REL = "benchmarks/mlsys_improvement_loop_v1/proposals/proposals.json"
SCHEMA = "mlsys_improvement_loop_trial_v1"

# Body of the empty-control subject: same interface, no work. Its runs measure
# sandbox start, interpreter start, input generation hand-off and frame I/O.
OVERHEAD_SOURCE = '''"""Empty control for timing: same interface, no work."""


class _Outcome:
    __slots__ = ("ok", "reason", "new_text")

    def __init__(self):
        self.ok = False
        self.reason = "no_edit_block"
        self.new_text = None


_ONE = _Outcome()


def parse_blocks(completion):
    return []


def apply_blocks(original, blocks):
    return _ONE
'''


class ProtocolError(RuntimeError):
    """The frozen inputs differ from the pre-registered protocol (fail closed)."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Protocol with every frozen hash re-verified against the working tree."""
    proto = json.loads(path.read_text(encoding="utf-8"))
    bad = [rel for rel, want in proto["frozen_sha256"].items() if sha256_text(_read(rel)) != want]
    if bad:
        raise ProtocolError(f"frozen file changed: {', '.join(sorted(bad))}")
    proto["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return proto


def load_module_from_text(name: str, text: str) -> ModuleType:
    """Import ``text`` as a fresh module (registered so dataclasses resolve)."""
    spec = importlib.util.spec_from_loader(name, loader=None)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        exec(compile(text, f"<{name}>", "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


# --------------------------------------------------------------------------- proposals


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    source: str  # "scripted" | "llm" | "human"
    description: str
    patch: str

    def record(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "source": self.source,
                "description": self.description, "patch_sha256": sha256_text(self.patch),
                "patch_bytes": len(self.patch.encode())}


def append_ledger(ledger: Path, event: Mapping[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {"recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **event}
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def export_view(dest: Path) -> list[str]:
    """Everything a proposer may see: the module, public tests and the task."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / TARGET_REL, dest / MODULE_FILE)
    shutil.copyfile(ROOT / PUBLIC_TESTS_REL, dest / Path(PUBLIC_TESTS_REL).name)
    shutil.copyfile(ROOT / TASK_REL, dest / "task.md")
    return sorted(p.name for p in dest.iterdir())


# --------------------------------------------------------------------------- stage 1: admission

_FORBIDDEN_HEADERS = ("new file mode", "deleted file mode", "old mode", "new mode", "rename from",
                      "rename to", "copy from", "copy to", "similarity index", "Binary files",
                      "GIT binary patch")


def _diff_path(raw: str) -> str:
    path = raw.split("\t", 1)[0].strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def admit(patch: str, pristine: str) -> dict[str, Any]:
    """Apply ``patch`` to a scratch copy; only an in-place edit of the module passes."""
    rec: dict[str, Any] = {"ok": False}
    touched: list[str] = []
    for line in patch.splitlines():
        if line.startswith(_FORBIDDEN_HEADERS):
            return {**rec, "reason": "forbidden_diff_entry", "detail": line[:80]}
        if line.startswith(("--- ", "+++ ")):
            touched.append(_diff_path(line[4:]))
        elif line.startswith("diff --git "):
            touched += [_diff_path(p) for p in line[len("diff --git "):].split(" ")]
    rec["touched"] = sorted(set(touched))
    if not touched:
        return {**rec, "reason": "not_a_diff"}
    if set(touched) != {MODULE_FILE}:
        return {**rec, "reason": "path_not_allowed"}
    with tempfile.TemporaryDirectory(prefix="mlsys-admit-") as tmp:
        work = Path(tmp)
        (work / MODULE_FILE).write_text(pristine, encoding="utf-8")
        before = sorted(p.name for p in work.iterdir())
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=work, input=patch.encode(),
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            return {**rec, "reason": "patch_does_not_apply",
                    "detail": r.stderr.decode("utf-8", "replace")[-300:]}
        if sorted(p.name for p in work.iterdir()) != before:
            return {**rec, "reason": "path_not_allowed"}
        candidate = (work / MODULE_FILE).read_text(encoding="utf-8")
    if candidate == pristine:
        return {**rec, "reason": "no_change"}
    try:
        compile(candidate, MODULE_FILE, "exec")
    except SyntaxError as exc:
        return {**rec, "reason": "syntax_error", "detail": str(exc)[:200]}
    return {**rec, "ok": True, "reason": "admitted", "candidate": candidate,
            "candidate_sha256": sha256_text(candidate)}


# --------------------------------------------------------------------------- stage 2: policy

ALLOWED_IMPORTS = frozenset({"__future__", "bisect", "collections", "dataclasses", "functools",
                             "itertools", "operator", "re", "string", "typing"})
DENIED_NAMES = frozenset({"__import__", "breakpoint", "compile", "delattr", "eval", "exec", "exit",
                          "getattr", "globals", "help", "input", "locals", "open", "quit",
                          "setattr", "vars"})
ALLOWED_DUNDERS = frozenset({"__all__", "__contains__", "__getitem__", "__init__", "__iter__",
                             "__len__", "__name__", "__post_init__", "__repr__", "__slots__"})


def policy_violations(source: str) -> list[str]:
    """Allowlist check over the whole candidate module (pristine passes)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["syntax_error"]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    found.add(f"import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                found.add(f"import:{'.' * node.level}{node.module or ''}")
        elif isinstance(node, ast.Name):
            if node.id in DENIED_NAMES:
                found.add(f"name:{node.id}")
            elif node.id.startswith("__") and node.id.endswith("__") and node.id not in ALLOWED_DUNDERS:
                found.add(f"dunder:{node.id}")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__") and attr not in ALLOWED_DUNDERS:
                found.add(f"dunder:{attr}")
            elif attr in DENIED_NAMES - {"compile"}:
                found.add(f"attr:{attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            continue
    for name in ("__builtins__", "sys.modules"):
        if name in source:
            found.add(f"text:{name}")
    return sorted(found)


# --------------------------------------------------------------------------- stage 4: tests


def test_ids(rel: str) -> list[str]:
    name = Path(rel).name
    return [f"{name}::{m}" for m in re.findall(r"^def (test_\w+)\(", _read(rel), flags=re.M)]


def stage_test_tree(dest: Path) -> list[str]:
    """Pristine module + public + held-out tests; returns the required test ids."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / TARGET_REL, dest / MODULE_FILE)
    for rel in (PUBLIC_TESTS_REL, HIDDEN_TESTS_REL):
        shutil.copyfile(ROOT / rel, dest / Path(rel).name)
    return [*test_ids(PUBLIC_TESTS_REL), *test_ids(HIDDEN_TESTS_REL)]


# --------------------------------------------------------------------------- sandbox benchmark runs


@dataclass(frozen=True)
class BenchRun:
    role: str
    status: str
    n: int
    digest: str
    ready_to_done_s: float
    wall_s: float
    inproc_s: float
    returncode: int
    error: str = ""


class Bench:
    """Runs the bench runner in the attested sandbox and reads frames live."""

    def __init__(self, backend: ig.BwrapBackend, workspace_root: Path, *, timeout: float = 180):
        self.backend = backend
        self.workspace_root = workspace_root
        self.timeout = timeout
        self.runner_source = _read(BENCH_RUNNER_REL)
        self.workload_source = _read(WORKLOAD_REL)

    def run(self, role: str, module_text: str, *, seed: int, count: int, profile: str) -> BenchRun:
        ws = Path(tempfile.mkdtemp(prefix="bench-", dir=self.workspace_root))
        try:
            (ws / MODULE_FILE).write_text(module_text, encoding="utf-8")
            return self._run(role, ws, seed=seed, count=count, profile=profile)
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def _run(self, role: str, ws: Path, *, seed: int, count: int, profile: str) -> BenchRun:
        key = secrets.token_bytes(32)
        nonce = secrets.randbits(62)
        spec = {"nonce": nonce, "count": count, "profile": profile, "module": MODULE,
                "import_roots": ["/work"], "workload_source": self.workload_source}
        cmd = [self.backend.python(), "-I", "-c", self.runner_source, json.dumps(spec)]
        stamps: dict[str, float] = {}
        frames: list[dict[str, Any]] = []
        err = tempfile.TemporaryFile()
        t0 = time.perf_counter()
        proc = subprocess.Popen(self.backend._argv(ws, cmd), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=err, start_new_session=True)
        killer = threading.Timer(self.timeout, proc.kill)
        killer.start()
        try:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write((json.dumps({"key": key.hex(), "seed": seed}) + "\n").encode())
            proc.stdin.close()
            raw_lines: list[tuple[float, bytes]] = []
            for line in proc.stdout:
                raw_lines.append((time.perf_counter(), line))
            rc = proc.wait()
            wall = time.perf_counter() - t0
        finally:
            killer.cancel()
            err.close()
        try:
            for when, raw in raw_lines:
                text = raw.decode("utf-8", "replace").rstrip("\n")
                if not text.startswith(tv.FRAME_PREFIX):
                    continue
                mac, _, payload = text[len(tv.FRAME_PREFIX):].partition(" ")
                want = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(mac, want):
                    raise tv.FrameError("bad_frame_mac")
                body = json.loads(payload)
                if body.get("nonce") != nonce or body.get("seq") != len(frames):
                    raise tv.FrameError("bad_frame_sequence")
                frames.append(body)
                stamps.setdefault(str(body.get("kind")), when)
            kinds = [f.get("kind") for f in frames]
            if kinds != ["start", "ready", "done", "result"]:
                raise tv.FrameError("incomplete_frames:" + ",".join(map(str, kinds)))
        except (tv.FrameError, json.JSONDecodeError) as exc:
            return BenchRun(role, "frame_error", 0, "", math.nan, wall, math.nan, rc, str(exc)[:120])
        done, result = frames[2], frames[3]
        status = str(done.get("status"))
        if status == "ok" and rc != 0:
            status = "exit_code_mismatch"
        return BenchRun(role, status, int(done.get("n", 0)), str(result.get("digest", "")),
                        stamps["done"] - stamps["ready"], wall, float(done.get("inproc_seconds", math.nan)),
                        rc)


# --------------------------------------------------------------------------- statistics


def speedup_stats(base: Sequence[float], cand: Sequence[float]) -> dict[str, Any]:
    """Geometric-mean speedup (baseline/candidate) with a paired 95% t interval."""
    logs = [math.log(b / c) for b, c in zip(base, cand)]
    mean, lo, hi, t, df = paired_difference_ci(logs)
    diff_mean, diff_lo, diff_hi, _, _ = paired_difference_ci([b - c for b, c in zip(base, cand)])
    return {"n": len(logs), "df": df, "t975": t,
            "speedup_point": math.exp(mean), "speedup_ci95": [math.exp(lo), math.exp(hi)],
            "seconds_saved_mean": diff_mean, "seconds_saved_ci95": [diff_lo, diff_hi],
            "log_ratio_sd": math.sqrt(sum((x - mean) ** 2 for x in logs) / (len(logs) - 1))}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------- the loop


@dataclass
class Session:
    proto: dict[str, Any]
    backend: ig.BwrapBackend
    attestation: ig.Attestation
    workspace_root: Path
    pristine: str
    pristine_mod: ModuleType
    workload: ModuleType
    bench: Bench
    cpu: int | None
    aa: dict[str, Any] | None = None

    @classmethod
    def open(cls, workspace_root: Path, evidence_dir: Path, *, proto: dict[str, Any] | None = None,
             venv: Path | None = None) -> Session:
        proto = proto or load_protocol()
        backend = ig.BwrapBackend(venv=venv or Path(sys.prefix))
        attestation = ig.attest(backend, workspace_root=workspace_root, evidence_dir=evidence_dir)
        pristine = _read(TARGET_REL)
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        return cls(proto, backend, attestation, workspace_root, pristine,
                   load_module_from_text("_mlsys_pristine_edit_format", pristine),
                   load_module_from_text("_mlsys_workload", _read(WORKLOAD_REL)),
                   Bench(backend, workspace_root), cpus[-1] if cpus else None)

    def expected_digest(self, seed: int, count: int, profile: str) -> str:
        cases = self.workload.make_cases(seed, count, profile)
        return str(self.workload.outcome_digest(self.workload.run_cases(self.pristine_mod, cases)))

    def environment(self) -> dict[str, Any]:
        return {"python": platform.python_version(), "platform": platform.platform(),
                "cpu_count": os.cpu_count(), "pinned_cpu": self.cpu,
                "loadavg_start": list(os.getloadavg()),
                "attestation": self.attestation.to_dict(),
                "config_digest": self.backend.config_digest(),
                "protocol_sha256": self.proto["_sha256"]}

    # -- timing ------------------------------------------------------------

    def measure(self, candidate: str, *, label: str) -> dict[str, Any]:
        m = self.proto["measurement"]
        count, profile = int(m["count"]), str(m["profile"])
        warmup, reps = int(m["warmup_reps"]), int(m["measured_reps"])
        rng = secrets.SystemRandom()
        roles = {"baseline": self.pristine, "candidate": candidate, "overhead": OVERHEAD_SOURCE}
        rows: list[dict[str, Any]] = []
        prev = os.sched_getaffinity(0) if self.cpu is not None else None
        if self.cpu is not None:
            os.sched_setaffinity(0, {self.cpu})
        try:
            for rep in range(warmup + reps):
                seed = secrets.randbits(63)
                expected = self.expected_digest(seed, count, profile)
                order = list(roles)
                rng.shuffle(order)
                runs = {r: self.bench.run(r, roles[r], seed=seed, count=count, profile=profile)
                        for r in order}
                rows.append({"rep": rep, "warmup": rep < warmup, "order": order,
                             "seed_sha256": sha256_text(str(seed)), "expected_digest": expected,
                             "runs": {r: asdict(v) for r, v in runs.items()},
                             "loadavg": list(os.getloadavg())})
        finally:
            if prev is not None:
                os.sched_setaffinity(0, prev)
        measured = [r for r in rows if not r["warmup"]]
        problems: list[str] = []
        for r in rows:
            for role in ("baseline", "candidate"):
                run = r["runs"][role]
                if run["status"] != "ok" or run["n"] != count:
                    problems.append(f"{role}_run_failed:{run['status']}")
                elif run["digest"] != r["expected_digest"]:
                    problems.append(f"{role}_output_mismatch")
            ov = r["runs"]["overhead"]
            if ov["status"] != "ok" or ov["n"] != count:
                problems.append(f"overhead_run_failed:{ov['status']}")
        out: dict[str, Any] = {"label": label, "count": count, "profile": profile,
                               "warmup_reps": warmup, "measured_reps": reps, "rows": rows,
                               "problems": sorted(set(problems))}
        if problems:
            return out
        base = [r["runs"]["baseline"]["ready_to_done_s"] for r in measured]
        cand = [r["runs"]["candidate"]["ready_to_done_s"] for r in measured]
        over = [r["runs"]["overhead"]["ready_to_done_s"] for r in measured]
        out["primary"] = speedup_stats(base, cand)
        out["secondary_wall"] = speedup_stats([r["runs"]["baseline"]["wall_s"] for r in measured],
                                              [r["runs"]["candidate"]["wall_s"] for r in measured])
        out["diagnostic_inproc"] = speedup_stats(
            [r["runs"]["baseline"]["inproc_s"] for r in measured],
            [r["runs"]["candidate"]["inproc_s"] for r in measured])
        out["means_s"] = {"baseline": _mean(base), "candidate": _mean(cand), "overhead": _mean(over)}
        out["workload_covers_startup"] = workload_covers_startup(base, cand, over)
        return out

    def calibrate(self) -> dict[str, Any]:
        """A/A: the unmodified module against a byte-identical copy of itself."""
        res = self.measure(self.pristine, label="aa_calibration")
        lo, hi = res.get("primary", {}).get("speedup_ci95", [math.nan, math.nan])
        res["valid"] = (not res["problems"]) and lo <= 1.0 <= hi
        self.aa = res
        return res

    # -- one proposal ------------------------------------------------------

    def evaluate(self, proposal: Proposal, *, ledger: Path, enforce_policy: bool = True,
                 enforce_static: bool = True) -> dict[str, Any]:
        rule = self.proto["acceptance"]
        audit = not (enforce_policy and enforce_static)
        append_ledger(ledger, {"event": "proposal_recorded", **proposal.record()})
        rec: dict[str, Any] = {"schema": SCHEMA, "proposal": proposal.record(), "stages": {},
                               "audit_mode": audit, "enforce_policy": enforce_policy,
                               "enforce_static": enforce_static,
                               "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "environment": self.environment()}
        t0 = time.time()

        def finish(decision: str, reason: str) -> dict[str, Any]:
            if audit and decision == "ACCEPT":
                rec["audit_decision"] = decision
                decision, reason = "REJECT", "audit_mode_never_accepts"
            rec.update(decision=decision, reason=reason, seconds=round(time.time() - t0, 2))
            append_ledger(ledger, {"event": "decision", "proposal_id": proposal.proposal_id,
                                   "patch_sha256": rec["proposal"]["patch_sha256"],
                                   "decision": decision, "reason": reason, "audit_mode": audit})
            return rec

        if not self.attestation.ok:
            return finish("REJECT", "isolation_unverified")
        adm = admit(proposal.patch, self.pristine)
        candidate = adm.pop("candidate", None)
        rec["stages"]["admission"] = adm
        if not adm["ok"] or candidate is None:
            return finish("REJECT", "admission:" + adm["reason"])
        viol = policy_violations(candidate)
        rec["stages"]["policy"] = {"violations": viol, "enforced": enforce_policy}
        if viol and enforce_policy:
            return finish("REJECT", "policy:" + viol[0])
        flags = tv.static_flags_v2(self.pristine, candidate)
        rec["stages"]["static"] = {"flags": flags, "enforced": enforce_static}
        if flags and enforce_static:
            return finish("REJECT", "static:" + flags[0])
        with tempfile.TemporaryDirectory(prefix="mlsys-tree-", dir=self.workspace_root) as tree:
            ids = stage_test_tree(Path(tree))
            grade = tv.grade_isolated_v2(
                self.backend, self.attestation, workspace_root=self.workspace_root,
                source_tree=Path(tree), target_file=MODULE_FILE, buggy_text=self.pristine,
                candidate_text=candidate, fail_to_pass=[], pass_to_pass=ids, import_roots=["/work"],
                timeout=float(rule["test_timeout_s"]), static_scan=enforce_static)
        rec["stages"]["tests"] = grade
        if not grade.get("passed"):
            return finish("REJECT", "tests:" + str(grade.get("reason")))
        d = self.proto["differential"]
        seed = secrets.randbits(63)
        expected = self.expected_digest(seed, int(d["count"]), str(d["profile"]))
        run = self.bench.run("differential", candidate, seed=seed, count=int(d["count"]),
                             profile=str(d["profile"]))
        same = run.status == "ok" and run.n == int(d["count"]) and run.digest == expected
        rec["stages"]["differential"] = {"run": asdict(run), "expected_digest": expected,
                                         "seed_sha256": sha256_text(str(seed)), "match": same}
        if not same:
            return finish("REJECT", "differential:" + (run.status if run.status != "ok" else "output_mismatch"))
        timing = self.measure(candidate, label=proposal.proposal_id)
        rec["stages"]["timing"] = timing
        if timing["problems"]:
            return finish("REJECT", "timing:" + timing["problems"][0])
        if self.aa is None or not self.aa.get("valid"):
            return finish("REJECT", "noise_floor_invalid")
        if not timing["workload_covers_startup"]:
            return finish("REJECT", "workload_below_startup_floor")
        prim = timing["primary"]
        if not prim["speedup_ci95"][0] > float(rule["speedup_ci95_low_gt"]):
            return finish("REJECT", "no_significant_speedup")
        if not prim["speedup_point"] >= float(rule["speedup_point_min"]):
            return finish("REJECT", "speedup_below_minimum")
        return finish("ACCEPT", "all_checks_passed")


# --------------------------------------------------------------------------- CLI


def load_scripted() -> list[dict[str, Any]]:
    items = json.loads(_read(PROPOSALS_REL))
    base = (ROOT / PROPOSALS_REL).parent
    for it in items:
        it["patch"] = (base / it["patch_file"]).read_text(encoding="utf-8")
    return items


def _summary_row(trial: Mapping[str, Any]) -> dict[str, Any]:
    timing = trial.get("stages", {}).get("timing", {})
    prim = timing.get("primary", {})
    return {"proposal_id": trial["proposal"]["proposal_id"], "decision": trial["decision"],
            "reason": trial["reason"], "audit_mode": trial["audit_mode"],
            "speedup_point": prim.get("speedup_point"), "speedup_ci95": prim.get("speedup_ci95"),
            "means_s": timing.get("means_s")}


def run_suite(out: Path, *, only: Sequence[str] = ()) -> dict[str, Any]:
    proto = load_protocol()
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger.jsonl"
    items = [it for it in load_scripted() if not only or it["proposal_id"] in only]
    with tempfile.TemporaryDirectory(prefix="mlsys-loop-") as root:
        session = Session.open(Path(root), out, proto=proto)
        env = session.environment()
        aa = session.calibrate()
        (out / "aa_calibration.json").write_text(json.dumps(aa, indent=2, sort_keys=True) + "\n")
        trials = []
        for it in items:
            prop = Proposal(it["proposal_id"], it["source"], it["description"], it["patch"])
            trial = session.evaluate(prop, ledger=ledger,
                                     enforce_policy=it.get("enforce_policy", True),
                                     enforce_static=it.get("enforce_static", True))
            trial["preregistered_expectation"] = it["expected_decision"]
            trial["category"] = it["category"]
            (out / f"trial_{it['proposal_id']}.json").write_text(
                json.dumps(trial, indent=2, sort_keys=True) + "\n")
            trials.append(trial)
            print(json.dumps(_summary_row(trial)), flush=True)
    summary = {"schema": "mlsys_improvement_loop_summary_v1", "protocol_sha256": proto["_sha256"],
               "environment": env,
               "aa_calibration": {"valid": aa["valid"], **{k: aa.get(k) for k in ("primary", "means_s",
                                                                                     "problems")}},
               "trials": [{**_summary_row(t), "category": t["category"],
                           "preregistered_expectation": t["preregistered_expectation"],
                           "matches_expectation": t["decision"] == t["preregistered_expectation"]}
                          for t in trials]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("export-view")
    v.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--patch", type=Path, required=True)
    e.add_argument("--source", choices=("scripted", "llm", "human"), required=True)
    e.add_argument("--description", default="")
    e.add_argument("--proposal-id", default="")
    e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("suite")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--only", nargs="*", default=())
    a = ap.parse_args(argv)
    if a.cmd == "export-view":
        print("\n".join(export_view(a.out)))
        return 0
    if a.cmd == "suite":
        run_suite(a.out, only=a.only)
        return 0
    patch = a.patch.read_text(encoding="utf-8")
    pid = a.proposal_id or "p-" + sha256_text(patch)[:12]
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlsys-loop-") as root:
        session = Session.open(Path(root), a.out)
        session.calibrate()
        trial = session.evaluate(Proposal(pid, a.source, a.description, patch), ledger=a.out / "ledger.jsonl")
        trial["aa_calibration"] = {k: session.aa.get(k) for k in ("valid", "primary", "means_s")} \
            if session.aa else None
    (a.out / f"trial_{pid}.json").write_text(json.dumps(trial, indent=2, sort_keys=True) + "\n")
    print(json.dumps(_summary_row(trial)))
    return 0 if trial["decision"] == "ACCEPT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
