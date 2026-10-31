"""Training-curriculum and held-out split builder.

Deterministic given (pinned dataset revision, repository lock, VARE tree,
config). The committed manifests contain identifiers and hashes only; bug
patches and problem statements are re-read from the pinned dataset at run
time, and the run refuses to start if the rebuilt manifest hash differs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import urllib.request

from . import exclusions, sources
from .edit_format import parse_single_file_patch, reference_fix_blocks, apply_blocks


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_shards(source: sources.TaskSource, cache: Path) -> list[dict[str, Any]]:
    out = []
    dest = cache / "datasets" / source.source_id / source.revision
    dest.mkdir(parents=True, exist_ok=True)
    for shard, url in zip(source.shards, source.shard_urls()):
        path = dest / Path(shard).name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - pinned HF URL
            tmp.replace(path)
        out.append({"shard": shard, "path": str(path), "sha256": sha256_file(path),
                    "bytes": path.stat().st_size})
    return out


def load_rows(shards: Iterable[Mapping[str, Any]], repos: set[str]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # optional dependency

    rows: list[dict[str, Any]] = []
    for shard in shards:
        table = pq.read_table(shard["path"], columns=list(sources.SWE_SMITH_PY.columns))
        for row in table.to_pylist():
            if row["repo"] in repos:
                rows.append(row)
    return rows


def changed_line_count(patch: str) -> int:
    fp = parse_single_file_patch(patch)
    return sum(1 for h in fp.hunks for tag, _ in h if tag in "+-")


def prefilter(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Static filters that need no repository checkout. Returns (reason, info)."""
    patch = row["patch"]
    if patch.count("diff --git ") != 1:
        return "not_single_file", {}
    try:
        fp = parse_single_file_patch(patch)
    except ValueError:
        return "unsupported_patch", {}
    path = fp.path
    low = path.lower()
    if not low.endswith(".py") or "test" in low.split("/")[-1] or "/tests/" in "/" + low:
        return "target_not_library_python", {}
    n_changed = changed_line_count(patch)
    if n_changed > cfg["max_changed_lines"]:
        return "patch_too_large", {}
    f2p, p2p = list(row["FAIL_TO_PASS"] or []), list(row["PASS_TO_PASS"] or [])
    if not 1 <= len(f2p) <= cfg["max_fail_to_pass"]:
        return "fail_to_pass_count", {}
    if len(p2p) > cfg["max_pass_to_pass"]:
        return "pass_to_pass_count", {}
    ps = (row.get("problem_statement") or "").strip()
    if not ps or len(ps) > cfg["max_problem_chars"]:
        return "problem_statement_length", {}
    return None, {"target_file": path, "changed_lines": n_changed}


def order_key(seed: str, instance_id: str) -> str:
    return sha256_text(f"{seed}:{instance_id}")


def build(
    *,
    cfg: Mapping[str, Any],
    repo_lock: Mapping[str, Any],
    vare_root: Path,
    cache: Path,
    out_dir: Path,
    materialize: bool = True,
) -> dict[str, Any]:
    """Build train/held-out manifests. Returns the build report."""
    from .repo_env import RepoEnv

    train_repos = list(cfg["train_repositories"])
    heldout_repos = list(cfg["heldout_repositories"])
    if set(train_repos) & set(heldout_repos):
        raise ValueError("train and held-out repositories must be disjoint")
    scan = exclusions.scan_vare_tree(vare_root, skip=cfg.get("scan_skip", ()))
    missing = exclusions.check_required(scan)
    if missing:
        raise RuntimeError(f"exclusion scan did not find required identifiers: {missing}")
    excluded = exclusions.excluded_repositories(scan)

    repo_report: dict[str, Any] = {}
    admitted_mirrors: dict[str, str] = {}
    for upstream in train_repos + heldout_repos:
        lock = repo_lock[upstream]
        lic = sources.REPO_LICENSES.get(upstream)
        reason = exclusions.is_excluded(upstream, excluded)
        if reason is None and not sources.is_permissive(lic):
            reason = "non_permissive_or_unknown_license"
        repo_report[upstream] = {"license": lic, "mirror": lock["mirror"], "excluded": reason}
        if reason is None:
            admitted_mirrors[lock["mirror"]] = upstream
    if any(v["excluded"] for v in repo_report.values()):
        bad = {k: v["excluded"] for k, v in repo_report.items() if v["excluded"]}
        raise RuntimeError(f"configured repositories are not admissible: {bad}")

    shards = fetch_shards(sources.SWE_SMITH_PY, cache)
    rows = load_rows(shards, set(admitted_mirrors))
    rejected: dict[str, int] = {}
    seen_patch: set[str] = set()
    seen_file_ps: set[tuple[str, str, str]] = set()
    candidates: dict[str, list[dict[str, Any]]] = {r: [] for r in admitted_mirrors.values()}
    for row in sorted(rows, key=lambda r: r["instance_id"]):
        reason, info = prefilter(row, cfg)
        if reason is None:
            ph = exclusions.normalized_patch_sha256(row["patch"])
            key = (row["repo"], info["target_file"], sha256_text(row["problem_statement"].strip()))
            if ph in seen_patch or key in seen_file_ps:
                reason = "duplicate"
            else:
                seen_patch.add(ph)
                seen_file_ps.add(key)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        upstream = admitted_mirrors[row["repo"]]
        candidates[upstream].append(
            {
                "instance_id": row["instance_id"],
                "upstream_repo": upstream,
                "upstream_license": sources.REPO_LICENSES[upstream],
                "source": sources.SWE_SMITH_PY.source_id,
                "source_revision": sources.SWE_SMITH_PY.revision,
                "dataset_license": sources.SWE_SMITH_PY.dataset_license,
                "mirror": row["repo"],
                "mirror_commit": repo_lock[upstream]["commit"],
                "target_file": info["target_file"],
                "changed_lines": info["changed_lines"],
                "patch_sha256": sha256_text(row["patch"]),
                "problem_statement_sha256": sha256_text(row["problem_statement"]),
                "fail_to_pass": list(row["FAIL_TO_PASS"]),
                # PASS_TO_PASS lists are long; the manifest keeps a hash and
                # attach_payloads() re-reads them from the pinned dataset.
                "pass_to_pass_count": len(row["PASS_TO_PASS"] or []),
                "pass_to_pass_sha256": sha256_text(json.dumps(list(row["PASS_TO_PASS"] or []))),
                "_pass_to_pass": list(row["PASS_TO_PASS"] or []),
                "order_key": order_key(cfg["seed"], row["instance_id"]),
                "_patch": row["patch"],
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "heldout": []}
    file_rejects: dict[str, int] = {}
    for split, repos, quota in (
        ("train", train_repos, cfg["train_per_repository"]),
        ("heldout", heldout_repos, cfg["heldout_per_repository"]),
    ):
        for upstream in repos:
            pool = sorted(candidates[upstream], key=lambda t: t["order_key"])
            env = RepoEnv(repo_lock[upstream]["mirror"], upstream, repo_lock[upstream]["commit"], cache)
            if materialize:
                env.materialize()
            taken = 0
            for task in pool:
                if taken >= quota:
                    break
                if materialize:
                    why = _file_check(env, task, cfg)
                    if why:
                        file_rejects[why] = file_rejects.get(why, 0) + 1
                        continue
                task["split"] = split
                selected[split].append(task)
                taken += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, tasks in selected.items():
        path = out_dir / f"{split}.manifest.jsonl"
        lines = [json.dumps({k: v for k, v in t.items() if not k.startswith("_")}, sort_keys=True)
                 for t in tasks]
        path.write_text("\n".join(lines) + "\n")
        hashes[split] = sha256_file(path)
    overlap = {t["instance_id"] for t in selected["train"]} & {t["instance_id"] for t in selected["heldout"]}
    report = {
        "source": {
            "source_id": sources.SWE_SMITH_PY.source_id,
            "hf_dataset": sources.SWE_SMITH_PY.hf_dataset,
            "revision": sources.SWE_SMITH_PY.revision,
            "dataset_license": sources.SWE_SMITH_PY.dataset_license,
            "citation": sources.SWE_SMITH_PY.citation,
            "shards": shards,
        },
        "license_only_sources": [
            {"source_id": s.source_id, "hf_dataset": s.hf_dataset, "revision": s.revision,
             "dataset_license": s.dataset_license, "execution": s.execution, "notes": s.notes}
            for s in sources.SOURCES.values() if s.execution != "native_venv"
        ],
        "repositories": repo_report,
        "exclusion_scan": {
            "files_scanned": scan["files_scanned"],
            "identifier_count": len(scan["identifiers"]),
            "required_identifiers_found": list(exclusions.REQUIRED_IDENTIFIERS),
            "excluded_repositories": sorted(excluded),
            "excluded_owners": sorted(exclusions.PROTECTED_OWNERS),
        },
        "rows_considered": len(rows),
        "rejected": dict(sorted(rejected.items())),
        "file_rejects": dict(sorted(file_rejects.items())),
        "counts": {k: len(v) for k, v in selected.items()},
        "per_repository": {
            split: {r: sum(t["upstream_repo"] == r for t in tasks) for r in sorted({t["upstream_repo"] for t in tasks})}
            for split, tasks in selected.items()
        },
        "train_heldout_instance_overlap": sorted(overlap),
        "manifest_sha256": hashes,
    }
    if overlap:
        raise RuntimeError(f"train/held-out overlap: {sorted(overlap)}")
    return report


def _file_check(env, task: Mapping[str, Any], cfg: Mapping[str, Any]) -> str | None:
    try:
        buggy = env.buggy_file_text(task["_patch"], task["target_file"])
    except Exception:  # noqa: BLE001 - recorded as a typed reject
        return "bug_patch_does_not_apply"
    if len(buggy) > cfg["max_file_chars"]:
        return "target_file_too_long"
    _, blocks = reference_fix_blocks(task["_patch"])
    res = apply_blocks(buggy, blocks)
    if not res.ok:
        return "reference_blocks_do_not_apply"
    if res.new_text != env.read(task["target_file"]):
        return "reference_blocks_do_not_restore_file"
    return None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attach_payloads(tasks: list[dict[str, Any]], cache: Path) -> list[dict[str, Any]]:
    """Re-read bug patch and problem statement from the pinned dataset and
    verify them against the manifest hashes. Fails closed on any mismatch."""
    shards = fetch_shards(sources.SWE_SMITH_PY, cache)
    want = {t["instance_id"]: t for t in tasks}
    rows = load_rows(shards, {t["mirror"] for t in tasks})
    found = {r["instance_id"]: r for r in rows if r["instance_id"] in want}
    out = []
    for iid, task in want.items():
        row = found.get(iid)
        if row is None:
            raise RuntimeError(f"{iid} missing from pinned dataset")
        if sha256_text(row["patch"]) != task["patch_sha256"]:
            raise RuntimeError(f"{iid} patch hash mismatch")
        if sha256_text(row["problem_statement"]) != task["problem_statement_sha256"]:
            raise RuntimeError(f"{iid} problem statement hash mismatch")
        p2p = list(row["PASS_TO_PASS"] or [])
        if sha256_text(json.dumps(p2p)) != task["pass_to_pass_sha256"]:
            raise RuntimeError(f"{iid} PASS_TO_PASS hash mismatch")
        if list(row["FAIL_TO_PASS"]) != task["fail_to_pass"]:
            raise RuntimeError(f"{iid} FAIL_TO_PASS mismatch")
        out.append({**task, "pass_to_pass": p2p, "_patch": row["patch"], "_problem": row["problem_statement"]})
    return out
