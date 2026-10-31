"""Run-002 held-out freeze: seeded selection, 26 x 8 floor and integrity checks on
synthetic build-check reports (instance IDs only; no task content)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run002_heldout_common as common  # noqa: E402

_spec = importlib.util.spec_from_file_location("run002_freeze_heldout",
                                               ROOT / "scripts" / "run002_freeze_heldout.py")
assert _spec and _spec.loader
fz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fz)

CANDS = common.load_candidates(min_passers=8)


def _task(iid: str, valid: bool) -> dict[str, Any]:
    h = common.task_order_key(iid)
    return {"instance_id": iid, "order_key": h, "valid": valid, "reason": "valid" if valid else "reference_fails",
            "fail_to_pass_count": 1, "pass_to_pass_count": 2,
            "hashes": {"patch_sha256": h, "problem_statement_sha256": h,
                       "fail_to_pass_sha256": h, "pass_to_pass_sha256": h}}


def synthetic_report(passing: set[str], *, r7_host: bool = True, invalid_first: int = 1) -> dict[str, Any]:
    repos = {}
    for c in CANDS:
        if c.upstream in passing:
            tasks = [_task(i, False) for i in c.passers[:invalid_first]]
            tasks += [_task(i, True) for i in c.passers[invalid_first:]][:8]
            status = "admitted" if sum(t["valid"] for t in tasks) >= 8 else "insufficient_valid_tasks"
        else:
            tasks, status = [], "env_failed"
        repos[c.upstream] = {"mirror": c.mirror, "commit": c.commit, "status": status, "tasks": tasks}
    return {"schema": fz.BUILD_CHECK_SCHEMA, "seed": common.SEED, "criteria_sha256": common.CRITERIA_SHA256,
            "inputs": common.verify_inputs(), "settings": dict(fz.R7_SETTINGS, grade_timeout_s=180.0,
                                                               min_passers=8),
            "environment": {"system": "Darwin" if r7_host else "Linux", "r7_host": r7_host}, "repos": repos}


def admittable(n: int) -> set[str]:
    """n repos that can be admitted with 8 tasks after one invalid task (>= 9 passers)."""
    ok = [c.upstream for c in CANDS if len(c.passers) >= 9]
    assert len(ok) >= n
    return set(ok[:n])


def run(rep: dict[str, Any], out: Path, **kw: Any) -> int:
    return fz.freeze(rep, out, excluded=lambda: set(), **kw)


def test_floor_is_single_named_constant() -> None:
    assert (fz.FLOOR_REPOS, fz.TASKS_PER_REPO, fz.MAX_REPOS) == (26, 8, 32)
    assert fz.build_parser().parse_args(["--report", "x"]).floor_repos == fz.FLOOR_REPOS
    assert len(CANDS) == 35


def test_cap_32x8_seeded_and_disjoint(tmp_path: Path) -> None:
    passing = admittable(33)
    assert run(synthetic_report(passing), tmp_path / "o") == fz.EXIT_OK
    rec = json.loads((tmp_path / "o" / "freeze.json").read_text())
    rows = [json.loads(x) for x in (tmp_path / "o" / "heldout.manifest.jsonl").read_text().splitlines()]
    assert rec["design"]["repos"] == 32 and rec["design"]["tasks"] == 256 == len(rows)
    assert rec["design"]["floor_repos"] == 26 and rec["design"]["floor_tasks"] == 208
    expected = [c.upstream for c in CANDS if c.upstream in passing][:32]
    assert [r["upstream"] for r in rec["repos"]] == expected
    keys = [common.repo_order_key(u) for u in expected]
    assert keys == sorted(keys)
    by = {c.upstream: c for c in CANDS}
    for r in rec["repos"]:
        assert r["instance_ids"] == list(by[r["upstream"]].passers[1:9])  # first invalid skipped
    assert all(v["overlap"] == 0 for v in rec["disjointness"].values())
    assert rec["synthetic"] is False and rec["frozen"] is True
    assert not any(k in rows[0] for k in ("fail_to_pass", "pass_to_pass", "patch", "problem_statement"))


@pytest.mark.parametrize("n", [26, 29, 31])
def test_between_floor_and_cap_takes_all_qualifying(tmp_path: Path, n: int) -> None:
    assert run(synthetic_report(admittable(n)), tmp_path / "o") == fz.EXIT_OK
    rec = json.loads((tmp_path / "o" / "freeze.json").read_text())
    assert rec["design"]["repos"] == n and rec["design"]["tasks"] == 8 * n and rec["design"]["shrunk_from_cap"]


def test_below_floor_refuses(tmp_path: Path) -> None:
    assert run(synthetic_report(admittable(25)), tmp_path / "o") == fz.EXIT_REFUSED
    assert not (tmp_path / "o" / "heldout.manifest.jsonl").exists()
    rec = json.loads((tmp_path / "o" / "refusal.json").read_text())
    assert rec["frozen"] is False and rec["design"]["qualifying_repos"] == 25


def test_floor_parameter_recorded(tmp_path: Path) -> None:
    assert run(synthetic_report(admittable(25)), tmp_path / "o", floor_repos=24) == fz.EXIT_OK
    rec = json.loads((tmp_path / "o" / "freeze.json").read_text())
    assert rec["design"]["floor_repos"] == 24 and rec["design"]["repos"] == 25


def test_no_seven_task_design(tmp_path: Path) -> None:
    # 30 repos with 7 valid tasks only: never frozen as 30 x 7
    rep = synthetic_report(admittable(30))
    for rec in rep["repos"].values():
        if rec["tasks"]:
            rec["tasks"][-1]["valid"] = False
            rec["status"] = "insufficient_valid_tasks"
    assert run(rep, tmp_path / "o") == fz.EXIT_REFUSED


def test_non_darwin_refused_unless_flagged(tmp_path: Path) -> None:
    rep = synthetic_report(admittable(26), r7_host=False)
    with pytest.raises(fz.FreezeError):
        run(rep, tmp_path / "a")
    assert run(rep, tmp_path / "b", allow_non_darwin=True) == fz.EXIT_OK
    assert json.loads((tmp_path / "b" / "freeze.json").read_text())["synthetic"] is True


def test_prefix_violation(tmp_path: Path) -> None:
    rep = synthetic_report(admittable(26))
    tasks = next(r["tasks"] for r in rep["repos"].values() if r["tasks"])
    tasks[0], tasks[1] = tasks[1], tasks[0]
    with pytest.raises(fz.FreezeError, match="prefix"):
        run(rep, tmp_path / "o")


def test_incomplete_report_and_bad_settings(tmp_path: Path) -> None:
    rep = synthetic_report(admittable(26))
    rep["repos"].pop(CANDS[0].upstream)
    with pytest.raises(fz.FreezeError, match="not finalized"):
        run(rep, tmp_path / "o")
    rep = synthetic_report(admittable(26))
    rep["settings"]["tasks_per_repo"] = 7
    with pytest.raises(fz.FreezeError, match="settings"):
        run(rep, tmp_path / "o")


def test_disjointness_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rep = synthetic_report(admittable(26))
    first = next(r["tasks"][1]["instance_id"] for r in rep["repos"].values() if r["tasks"])
    real = common.exposed_ids
    monkeypatch.setattr(common, "exposed_ids", lambda d=common.INPUT_DIR: {**real(d), "exposure_ledger": {first}})
    with pytest.raises(fz.FreezeError, match="overlap"):
        run(rep, tmp_path / "o")


def test_excluded_repo_and_no_overwrite(tmp_path: Path) -> None:
    rep = synthetic_report(admittable(26))
    with pytest.raises(fz.FreezeError, match="exclusion"):
        fz.freeze(rep, tmp_path / "o", excluded=lambda: {CANDS[0].upstream})
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "x").write_text("")
    with pytest.raises(fz.FreezeError, match="overwrite"):
        run(rep, tmp_path / "p")


def test_cli_exit_codes(tmp_path: Path) -> None:
    rep_path = tmp_path / "r.json"
    rep_path.write_text(json.dumps(synthetic_report(admittable(26), r7_host=False)))
    assert fz.main(["--report", str(rep_path), "--out", str(tmp_path / "a")]) == fz.EXIT_INPUT
    rep_path.write_text(json.dumps(synthetic_report(admittable(20))))
    assert fz.main(["--report", str(rep_path), "--out", str(tmp_path / "b")]) == fz.EXIT_REFUSED
