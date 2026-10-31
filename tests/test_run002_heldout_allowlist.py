"""Run-002 held-out allowlist extension: well-formedness, licence evidence, and proof that the
extension leaves the frozen run-001 inputs and the v1 Train / held-out selection unchanged.
IDs, counts and hashes only; no network."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from scripts import run002_heldout_common as common
from vare.learning_pipeline import curriculum, exclusions, sources

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs" / "learning_pipeline_v1"
PROTOCOL_V1 = ROOT / "protocols" / "coding_policy_learning_mlx_4b_v1.json"
NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# sha256 of json.dumps(sorted(sources.REPO_LICENSES.items())) at the run-001 freeze.
V1_ALLOWLIST_SHA256 = "529a4914f48a10a3b0313e47cba6665d4700d01d844aedec720f428afa0d3070"


def _counts() -> dict:
    return json.loads((common.INPUT_DIR / common.COUNTS_FILE).read_text())


def _criteria_group_b() -> dict[str, tuple[str, str]]:
    text = (common.INPUT_DIR / common.CRITERIA_FILE).read_text()
    rows = re.findall(r"^\| (\S+/\S+) \| (\S+) \| (github_api|rebench_v2_field) \|$", text, re.M)
    return {repo: (lic, src) for repo, lic, src in rows}


def test_inputs_match_pins_and_sha256sums():
    assert common.verify_inputs() == common.INPUT_SHA256
    sums = {}
    for line in (common.INPUT_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    assert sums == common.INPUT_SHA256
    assert common.CRITERIA_SHA256.startswith("7ed490f8")
    assert _counts()["criteria_sha256"] == common.CRITERIA_SHA256


def test_extension_entries_are_well_formed():
    ext, src = common.RUN002_REPO_LICENSES, common.RUN002_LICENSE_SOURCES
    assert len(ext) == 26 and set(ext) == set(src)
    train, held = common.run001_repositories()
    for repo, lic in ext.items():
        assert NAME.match(repo), repo
        assert lic in sources.PERMISSIVE_LICENSES, repo
        assert src[repo] in {"github_api", "rebench_v2_field"}, repo
        assert repo not in sources.REPO_LICENSES, repo
        assert repo.split("/")[0].lower() not in exclusions.PROTECTED_OWNERS, repo
        assert repo.lower() not in {r.lower() for r in exclusions.PROTECTED_REPOSITORIES}, repo
        assert repo not in train + held, repo
    merged = common.admitted_licenses()
    assert merged == {**sources.REPO_LICENSES, **ext} and len(merged) == len(sources.REPO_LICENSES) + 26


def test_extension_matches_recorded_licence_evidence():
    counts = _counts()["repos"]
    table = _criteria_group_b()
    assert len(table) == 45  # criteria section 1 group B, closed list
    for repo, lic in common.RUN002_REPO_LICENSES.items():
        rec = counts[repo]
        assert rec["group"] == "B", repo
        assert table[repo] == (lic, common.RUN002_LICENSE_SOURCES[repo]), repo
        assert rec["license_survey"] == [common.RUN002_LICENSE_SOURCES[repo], lic], repo
        # R1: the code-host licence API re-check must return the same permissive id,
        # which is mandatory for rebench_v2_field survey entries.
        assert rec["license_api_recheck"] == lic and rec["license_ok"] is True, repo
        assert rec["scan_excluded"] is None and rec["heavy_excluded"] is False, repo
        assert rec["admissible_pre_env"] is True, repo


def test_extension_is_exactly_the_admissible_group_b_set():
    counts = _counts()["repos"]
    passers = common.load_passers(common.INPUT_DIR / common.PASSERS_FILE)
    for k in (common.MIN_TASKS_PER_REPO, common.MAX_TASKS_PER_REPO):
        want = {r for r, rec in counts.items()
                if rec["group"] == "B" and rec["admissible_pre_env"] and len(passers.get(r, [])) >= k}
        assert want == set(common.RUN002_REPO_LICENSES), k
    # R5-excluded group B repos are never admitted, whatever their passer count.
    heavy = {r for r, rec in counts.items() if rec["heavy_excluded"]}
    assert heavy and not heavy & set(common.RUN002_REPO_LICENSES)


def test_candidate_pool_counts_and_group_a_licences():
    c8 = common.load_candidates(min_passers=8)
    c7 = common.load_candidates(min_passers=7)
    assert (len(c8), sum(c.group == "A" for c in c8)) == (35, 9)
    assert len(c7) == 36 and {c.upstream for c in c7} - {c.upstream for c in c8} == {"un33k/python-slugify"}
    for c in c7:
        assert re.fullmatch(r"[0-9a-f]{40}", c.commit), c.upstream
        assert sources.upstream_repo(c.mirror)[0] == c.upstream
        assert all(i.startswith(c.mirror.split("/", 1)[1] + ".") for i in c.passers)
        if c.group == "A":
            assert c.license == sources.REPO_LICENSES[c.upstream]
    keys = [c.repo_key for c in c7]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_exclusion_scan_admits_every_candidate_and_run001_repo():
    res = common.exclusion_scan(ROOT)
    assert res["required_missing"] == []
    train, held = common.run001_repositories()
    for repo in [c.upstream for c in common.load_candidates(min_passers=7)] + train + held:
        assert exclusions.is_excluded(repo, res["excluded"]) is None, repo


def test_run001_frozen_inputs_and_allowlist_unchanged():
    proto = json.loads(PROTOCOL_V1.read_text())
    for rel in ("configs/learning_pipeline_v1/curriculum.json", "configs/learning_pipeline_v1/repos.lock.json",
                "configs/learning_pipeline_v1/train.manifest.jsonl",
                "configs/learning_pipeline_v1/heldout.manifest.jsonl"):
        assert curriculum.sha256_file(ROOT / rel) == proto["frozen"]["inputs"][rel], rel
    rel = "src/vare/learning_pipeline/sources.py"
    assert curriculum.sha256_file(ROOT / rel) == proto["frozen"]["code"][rel]
    digest = hashlib.sha256(json.dumps(sorted(sources.REPO_LICENSES.items())).encode()).hexdigest()
    assert digest == V1_ALLOWLIST_SHA256
    exposed = common.exposed_ids()
    assert len(exposed["train_v1"]) == 80 and len(exposed["heldout_run001"]) == 42
    assert exposed["train_v1"] | exposed["heldout_run001"] <= exposed["exposure_ledger"]
    pool = {i for c in common.load_candidates(min_passers=7) for i in c.passers}
    assert not pool & set().union(*exposed.values())


def _fake_rows(mirror: str, n: int) -> list[dict]:
    rows = []
    tag = hashlib.sha256(mirror.encode()).hexdigest()[:8]
    for i in range(n):
        patch = (f"diff --git a/pkg/m{i}.py b/pkg/m{i}.py\n--- a/pkg/m{i}.py\n+++ b/pkg/m{i}.py\n"
                 f"@@ -1,2 +1,2 @@\n def f{i}_{tag}(a):\n-    return a + {i}\n+    return a - {i}\n")
        rows.append({"instance_id": f"{mirror.split('/', 1)[1]}.fake_{i:03d}", "repo": mirror, "patch": patch,
                     "FAIL_TO_PASS": [f"tests/test_m.py::test_{i}"], "PASS_TO_PASS": [],
                     "problem_statement": f"synthetic statement {mirror} {i}"})
    return rows


def _build(tmp_path: Path, tag: str, rows: list[dict], monkeypatch) -> tuple[dict, dict[str, bytes]]:
    cfg = json.loads((V1 / "curriculum.json").read_text())
    lock = json.loads((V1 / "repos.lock.json").read_text())
    root = tmp_path / "tree"
    root.mkdir(exist_ok=True)
    (root / "refs.md").write_text(" ".join(exclusions.REQUIRED_IDENTIFIERS) + "\n")
    monkeypatch.setattr(curriculum, "fetch_shards", lambda source, cache: [])
    monkeypatch.setattr(curriculum, "load_rows", lambda shards, repos: [r for r in rows if r["repo"] in repos])
    out = tmp_path / tag
    report = curriculum.build(cfg=cfg, repo_lock=lock, vare_root=root, cache=tmp_path / "cache", out_dir=out,
                              materialize=False)
    return report, {p.name: p.read_bytes() for p in sorted(out.glob("*.jsonl"))}


def test_extension_does_not_change_v1_train_or_heldout_selection(tmp_path, monkeypatch):
    lock = json.loads((V1 / "repos.lock.json").read_text())
    rows = [r for up in lock for r in _fake_rows(lock[up]["mirror"], 20)]
    new = {c.mirror for c in common.load_candidates(min_passers=7)}
    for mirror in sorted(new)[:5]:  # rows of new repos are present in the dataset
        rows += _fake_rows(mirror, 20)
    base_report, base = _build(tmp_path, "v1", rows, monkeypatch)
    monkeypatch.setattr(sources, "REPO_LICENSES", common.admitted_licenses())
    ext_report, ext = _build(tmp_path, "ext", rows, monkeypatch)
    assert base == ext and base_report == ext_report
    assert base_report["counts"] == {"train": 80, "heldout": 42}
    assert not any(json.loads(line)["mirror"] in new for blob in ext.values() for line in blob.splitlines())


@pytest.mark.parametrize("bad", [{"Suor/funcy": "MIT"}, {"theskumar/python-dotenv": "MIT"}])
def test_extension_may_not_override_v1_entries(monkeypatch, bad):
    monkeypatch.setattr(common, "RUN002_REPO_LICENSES", {**common.RUN002_REPO_LICENSES, **bad})
    with pytest.raises(ValueError):
        common.admitted_licenses()
