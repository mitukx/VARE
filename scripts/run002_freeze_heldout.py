#!/usr/bin/env python3
"""Run-002 held-out pool: freeze the manifest from a completed R7 build-check report.

Design rule (lead decision, k fixed at 8; no k = 7 design):
  * a repo qualifies when the report shows it admitted with ``TASKS_PER_REPO`` (8) tasks that
    validated, taken as the first validated tasks in ascending ``order_key(SEED, id)``;
  * qualifying repos are ordered by sha256 of ``"{SEED}:repo:{upstream}"`` and the first
    ``min(MAX_REPOS, n)`` are taken (32 x 8 when >= 32 qualify, else n x 8);
  * when fewer than ``FLOOR_REPOS`` (26) repos qualify nothing is frozen: a refusal record is
    written and the script exits 3.

Integrity checks (exit 2 on failure): frozen inputs and criteria hash; report schema, seed,
inputs and R7 settings; macOS host (``--allow-non-darwin`` marks the output synthetic); every
candidate repo finalized; attempted tasks are exactly a prefix of the seeded passer order;
licence admitted; not a run-001 repo; not excluded by the repository exclusion scan; selected
IDs disjoint from Train v1, run-001 held-out and the exposure ledger. Existing output is never
overwritten.

Outputs carry instance IDs, repos, seeds and hashes only: no task content.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run002_heldout_common as common  # noqa: E402

from vare.learning_pipeline import sources  # noqa: E402

SCHEMA = "vare.run002.heldout_freeze.v1"
BUILD_CHECK_SCHEMA = "vare.run002.build_check.v1"
TASKS_PER_REPO = 8  # fixed design k (lead decision); no k = 7 fallback
MAX_REPOS = common.MAX_REPOS  # 32
FLOOR_REPOS = 26  # refusal floor: fewer than FLOOR_REPOS x TASKS_PER_REPO is not frozen
R7_SETTINGS = {"install_timeout_s": 900, "venv_limit_kib": 1024 * 1024, "clone_limit_bytes": 300_000_000,
               "max_attempts_per_repo": 0, "tasks_per_repo": TASKS_PER_REPO}
FINAL_STATUSES = {"admitted", "insufficient_valid_tasks", "attempt_cap_reached", "env_failed",
                  "size_exceeded", "payload_error"}
EXIT_OK, EXIT_INPUT, EXIT_REFUSED = 0, 2, 3
DEFAULT_OUT = Path("artifacts/learning_pipeline_run002/heldout_freeze")


class FreezeError(Exception):
    """Integrity failure; nothing is frozen (exit 2)."""


def check_report(rep: dict[str, Any], inputs: dict[str, str], allow_non_darwin: bool) -> bool:
    """Validate report header; returns True when the report is not R7 evidence (synthetic)."""
    if rep.get("schema") != BUILD_CHECK_SCHEMA:
        raise FreezeError("report schema mismatch")
    if rep.get("seed") != common.SEED or rep.get("criteria_sha256") != common.CRITERIA_SHA256:
        raise FreezeError("report seed or criteria hash differs from the frozen inputs")
    if rep.get("inputs") != inputs:
        raise FreezeError("report input hashes differ from the frozen inputs")
    settings = rep.get("settings") or {}
    bad = sorted(k for k, v in R7_SETTINGS.items() if settings.get(k) != v)
    if bad:
        raise FreezeError(f"report settings differ from R7: {', '.join(bad)}")
    r7_host = bool((rep.get("environment") or {}).get("r7_host"))
    if not r7_host and not allow_non_darwin:
        raise FreezeError("report is not from a macOS host (not R7 evidence); --allow-non-darwin for tests")
    return not r7_host


def repo_tasks(cand: common.Candidate, rec: dict[str, Any] | None) -> list[str] | None:
    """The first TASKS_PER_REPO validated IDs, or None if the repo does not qualify."""
    if rec is None or rec.get("status") not in FINAL_STATUSES:
        raise FreezeError(f"{cand.upstream}: not finalized in the report (resume the build check)")
    if rec.get("mirror") != cand.mirror or rec.get("commit") != cand.commit:
        raise FreezeError(f"{cand.upstream}: mirror or commit differs from the frozen inputs")
    attempted = [t["instance_id"] for t in rec.get("tasks") or []]
    if attempted != list(cand.passers[: len(attempted)]):
        raise FreezeError(f"{cand.upstream}: attempted tasks are not a prefix of the seeded order")
    valid = [t["instance_id"] for t in rec.get("tasks") or [] if t.get("valid")]
    if rec["status"] != "admitted":
        return None
    if len(valid) < TASKS_PER_REPO:
        raise FreezeError(f"{cand.upstream}: admitted with fewer than {TASKS_PER_REPO} valid tasks")
    return valid[:TASKS_PER_REPO]


def manifest_row(cand: common.Candidate, task: dict[str, Any]) -> dict[str, Any]:
    src = sources.SWE_SMITH_PY
    row = {"instance_id": task["instance_id"], "split": "heldout", "upstream_repo": cand.upstream,
           "upstream_license": cand.license, "license_source": cand.license_source, "group": cand.group,
           "source": src.source_id, "source_revision": src.revision, "dataset_license": src.dataset_license,
           "mirror": cand.mirror, "mirror_commit": cand.commit,
           "order_key": common.task_order_key(task["instance_id"]), "repo_order_key": cand.repo_key,
           "fail_to_pass_count": task.get("fail_to_pass_count"),
           "pass_to_pass_count": task.get("pass_to_pass_count")}
    row.update(task.get("hashes") or {})
    return row


def freeze(rep: dict[str, Any], out_dir: Path, *, floor_repos: int = FLOOR_REPOS,
           allow_non_darwin: bool = False, input_dir: Path = common.INPUT_DIR,
           excluded: Callable[[], set[str]] | None = None, report_sha256: str | None = None) -> int:
    if not 1 <= floor_repos <= MAX_REPOS:
        raise FreezeError(f"floor repos must be in 1..{MAX_REPOS}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FreezeError("output directory is not empty; refusing to overwrite")
    inputs = common.verify_inputs(input_dir)
    synthetic = check_report(rep, inputs, allow_non_darwin)
    cands = common.load_candidates(input_dir, min_passers=TASKS_PER_REPO)
    licenses = common.admitted_licenses()
    train_repos, heldout_repos = common.run001_repositories()
    excl = excluded() if excluded else set(common.exclusion_scan()["excluded"])

    qualifying: list[tuple[common.Candidate, list[str]]] = []
    status_counts: dict[str, int] = {}
    for cand in cands:  # already in seeded repo order
        rec = rep["repos"].get(cand.upstream)
        ids = repo_tasks(cand, rec)
        status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1  # type: ignore[index]
        if licenses.get(cand.upstream) != cand.license:
            raise FreezeError(f"{cand.upstream}: licence not on the admitted allowlist")
        if cand.upstream in train_repos or cand.upstream in heldout_repos:
            raise FreezeError(f"{cand.upstream}: run-001 repository")
        if cand.upstream in excl:
            raise FreezeError(f"{cand.upstream}: excluded by the repository exclusion scan")
        if ids is not None:
            qualifying.append((cand, ids))

    design = {"tasks_per_repo": TASKS_PER_REPO, "max_repos": MAX_REPOS, "floor_repos": floor_repos,
              "floor_tasks": floor_repos * TASKS_PER_REPO, "qualifying_repos": len(qualifying),
              "candidate_repos": len(cands)}
    record: dict[str, Any] = {"schema": SCHEMA, "seed": common.SEED, "criteria_sha256": common.CRITERIA_SHA256,
                              "inputs": inputs, "report_sha256": report_sha256, "synthetic": synthetic,
                              "design": design, "status_counts": dict(sorted(status_counts.items())),
                              "code_sha256": {Path(p).name: common.sha256_file(Path(p)) for p in
                                              (__file__, common.__file__)}}
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(qualifying) < floor_repos:
        record.update(frozen=False, refusal=(f"{len(qualifying)} repos qualify with {TASKS_PER_REPO} tasks; "
                                             f"floor is {floor_repos} x {TASKS_PER_REPO}"))
        common.atomic_write_json(out_dir / "refusal.json", record)
        print(f"REFUSED: {record['refusal']}", file=sys.stderr)
        return EXIT_REFUSED

    chosen = qualifying[:MAX_REPOS]
    rows: list[dict[str, Any]] = []
    for cand, ids in chosen:
        by_id = {t["instance_id"]: t for t in rep["repos"][cand.upstream]["tasks"]}
        rows += [manifest_row(cand, by_id[i]) for i in ids]
    selected = {r["instance_id"] for r in rows}
    exposed = common.exposed_ids(input_dir)
    overlap = {name: len(selected & ids) for name, ids in exposed.items()}
    if any(overlap.values()):
        raise FreezeError(f"selected IDs overlap exposed sets: {overlap}")
    rows.sort(key=lambda r: (r["repo_order_key"], r["order_key"]))
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (out_dir / "heldout.manifest.jsonl").write_text(text)
    design.update(repos=len(chosen), tasks=len(rows),
                  shrunk_from_cap=len(chosen) < MAX_REPOS)
    record.update(frozen=True, manifest_sha256=common.sha256_bytes(text.encode()),
                  repos=[{"upstream": c.upstream, "group": c.group, "repo_order_key": c.repo_key,
                          "instance_ids": ids} for c, ids in chosen],
                  disjointness={name: {"exposed": len(ids), "overlap": overlap[name]}
                                for name, ids in exposed.items()})
    common.atomic_write_json(out_dir / "freeze.json", record)
    print(f"frozen {len(chosen)} repos x {TASKS_PER_REPO} = {len(rows)} tasks"
          f"{' (SYNTHETIC: not R7 evidence)' if synthetic else ''}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="completed run002_build_check.py report (from the Mac)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="new, empty output directory")
    ap.add_argument("--floor-repos", type=int, default=FLOOR_REPOS,
                    help=f"refuse below this many repos x {TASKS_PER_REPO} tasks (default {FLOOR_REPOS})")
    ap.add_argument("--allow-non-darwin", action="store_true", help="tests only; output marked synthetic")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    report = Path(a.report)
    try:
        rep = json.loads(report.read_text())
        return freeze(rep, Path(a.out), floor_repos=a.floor_repos, allow_non_darwin=a.allow_non_darwin,
                      report_sha256=common.sha256_file(report))
    except (FreezeError, RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"error: {common.relativize(str(exc))}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
