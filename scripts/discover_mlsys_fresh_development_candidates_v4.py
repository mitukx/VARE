#!/usr/bin/env python3
"""Capture a one-day, metadata-only incremental pool of merged ML-systems PRs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_discovery_v4.json"
SPECIALIZED_TITLE_MARKERS = (
    "gpu", "metal", "cuda", "fsdp", "distributed", "expert parallel",
    "neuron", "architecture", "support ", "add ", "integration test",
    "preemption",
)
TEST_PATH_MARKERS = ("test", "spec", "fixture")
SEARCH_QUERY = """
query($searchText: String!, $cursor: String) {
  search(query: $searchText, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest { number title url state mergedAt }
    }
  }
}
"""
PULL_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number title url state isDraft createdAt updatedAt mergedAt
      baseRefOid headRefOid additions deletions changedFiles
      mergeCommit { oid parents(first: 2) { nodes { oid } } }
      files(first: 100) {
        pageInfo { hasNextPage }
        nodes { path }
      }
    }
  }
}
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_gh_graphql(query: str, variables: dict[str, str | int]) -> dict[str, Any]:
    arguments = ["gh", "api", "graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        flag = "-F" if type(value) is int else "-f"
        arguments.extend([flag, f"{name}={value}"])
    result = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=90)
    payload = json.loads(result.stdout)
    if type(payload) is not dict or payload.get("errors"):
        raise ValueError(f"GitHub GraphQL query failed: {(payload or {}).get('errors')}")
    return payload


def search_repository(repo: str, date_start: str, date_end: str) -> tuple[int, list[dict[str, Any]]]:
    search_text = f"repo:{repo} is:pr is:merged merged:{date_start}..{date_end}"
    page = run_gh_graphql(SEARCH_QUERY, {"searchText": search_text})["data"]["search"]
    total = page.get("issueCount")
    if type(total) is not int or total > 1000:
        raise ValueError(f"GitHub search count is missing or exceeds 1,000 for {repo}")
    nodes, info = page.get("nodes"), page.get("pageInfo")
    if type(nodes) is not list or type(info) is not dict:
        raise ValueError(f"GitHub search response is incomplete for {repo}")
    rows = list(nodes)
    while info.get("hasNextPage"):
        cursor = info.get("endCursor")
        if type(cursor) is not str:
            raise ValueError(f"GitHub search cursor is missing for {repo}")
        page = run_gh_graphql(
            SEARCH_QUERY, {"searchText": search_text, "cursor": cursor},
        )["data"]["search"]
        next_nodes, info = page.get("nodes"), page.get("pageInfo")
        if type(next_nodes) is not list or not next_nodes or type(info) is not dict:
            raise ValueError(f"GitHub search ended before its declared count for {repo}")
        rows.extend(next_nodes)
    if len(rows) != total:
        raise ValueError(f"GitHub search count differs from returned rows for {repo}")
    if any(type(row) is not dict or type(row.get("number")) is not int for row in rows):
        raise ValueError(f"GitHub returned a non-pull-request search row for {repo}")
    return total, rows


def pull_metadata(repo: str, number: int) -> dict[str, Any]:
    owner, name = repo.split("/", maxsplit=1)
    payload = run_gh_graphql(PULL_QUERY, {"owner": owner, "name": name, "number": number})
    pr = ((payload.get("data") or {}).get("repository") or {}).get("pullRequest")
    if type(pr) is not dict:
        raise ValueError(f"GitHub returned no pull request for {repo}#{number}")
    files = pr.get("files")
    nodes = files.get("nodes") if type(files) is dict else None
    if type(nodes) is not list or any(type(row) is not dict or type(row.get("path")) is not str for row in nodes):
        raise ValueError(f"GitHub returned malformed changed paths for {repo}#{number}")
    merge_commit = pr.get("mergeCommit")
    if merge_commit is not None and type(merge_commit) is not dict:
        raise ValueError(f"GitHub returned malformed merge metadata for {repo}#{number}")
    parents = (merge_commit or {}).get("parents", {}).get("nodes", [])
    return {
        "repository": repo,
        "number": pr.get("number"),
        "title": pr.get("title"),
        "url": pr.get("url"),
        "state": pr.get("state"),
        "is_draft": pr.get("isDraft"),
        "created_at": pr.get("createdAt"),
        "updated_at": pr.get("updatedAt"),
        "merged_at": pr.get("mergedAt"),
        "base_ref_sha_at_capture": pr.get("baseRefOid"),
        "head_sha": pr.get("headRefOid"),
        "merge_base_sha": parents[0].get("oid") if parents else None,
        "merge_commit_sha": merge_commit.get("oid") if merge_commit else None,
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_file_count": pr.get("changedFiles"),
        "changed_paths": [row["path"] for row in nodes],
        "changed_paths_complete": not files["pageInfo"].get("hasNextPage", False),
    }


def is_test_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in TEST_PATH_MARKERS)


def screen_candidate(
    row: dict[str, Any], *, protocol: dict[str, Any], prior: set[tuple[str, int]],
) -> dict[str, Any]:
    rules = protocol["coarse_screen"]
    discovery = protocol["discovery"]
    reasons: list[str] = []
    identity = (row.get("repository"), row.get("number"))
    merged_at = row.get("merged_at") or ""
    merged_date = merged_at[:10]
    if row.get("state") != "MERGED" or not merged_at or not row.get("merge_commit_sha"):
        reasons.append("not_merged_or_missing_merge_commit")
    if not row.get("merge_base_sha"):
        reasons.append("merge_base_commit_missing")
    if not discovery["merged_date_start"] <= merged_date <= discovery["merged_date_end"]:
        reasons.append("merged_outside_frozen_date_window")
    if merged_at > discovery["snapshot_cutoff_utc"]:
        reasons.append("merged_after_frozen_snapshot_cutoff")
    if merged_date <= discovery["model_training_cutoff_proxy"]:
        reasons.append("not_after_model_training_cutoff_proxy")
    if not row.get("changed_paths_complete"):
        reasons.append("changed_path_inventory_incomplete")
    paths = row.get("changed_paths") if type(row.get("changed_paths")) is list else []
    test_paths = [path for path in paths if type(path) is str and is_test_path(path)]
    if not test_paths:
        reasons.append("no_test_path_in_merged_pr_diff")
    additions, deletions = row.get("additions"), row.get("deletions")
    if type(additions) is not int or type(deletions) is not int:
        reasons.append("changed_line_count_missing")
    elif additions + deletions > rules["maximum_changed_lines"]:
        reasons.append("diff_exceeds_maximum_changed_lines")
    changed_file_count = row.get("changed_file_count")
    if type(changed_file_count) is not int:
        reasons.append("changed_file_count_missing")
    elif changed_file_count > rules["maximum_changed_files"]:
        reasons.append("diff_exceeds_maximum_changed_files")
    title = (row.get("title") or "").lower()
    if any(marker in title for marker in rules["specialized_title_markers"]):
        reasons.append("title_flags_specialized_or_feature_scope_for_manual_review")
    if identity in prior:
        reasons.append("exact_identity_already_in_development_train_or_prior_queue")
    return {
        **row,
        "test_paths": test_paths,
        "coarse_screen": "provisional_for_manual_review" if not reasons else "excluded_from_shortlist",
        "screen_reasons": reasons,
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
    }


def frozen_inputs(protocol: dict[str, Any]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for row in protocol["prior_exposure_inputs"]:
        result = subprocess.run(
            ["git", "show", f"{protocol['source_main_commit']}:{row['path']}"],
            check=True,
            capture_output=True,
        )
        if sha256_bytes(result.stdout) != row["sha256"]:
            raise ValueError(f"frozen exposure input changed: {row['path']}")
        payloads[row["path"]] = result.stdout
    if subprocess.run(
        ["git", "cat-file", "-e", f"{protocol['source_main_commit']}^{{commit}}"],
        check=False, capture_output=True,
    ).returncode:
        raise ValueError("frozen source main commit is unavailable")
    if sha256_file(Path(__file__)) != protocol["collector_sha256"]:
        raise ValueError("collector differs from the frozen protocol")
    return payloads


def load_prior_identities(protocol: dict[str, Any], inputs: dict[str, bytes]) -> set[tuple[str, int]]:
    prior: set[tuple[str, int]] = set()
    queue_bytes = inputs.get(protocol["prior_candidate_queue_path"])
    if queue_bytes is None:
        raise ValueError("prior candidate queue is not a hash-verified input")
    queue = json.loads(queue_bytes.decode("utf-8"))
    for row in queue.get("candidates", []):
        if row.get("historical_split") == "heldout":
            continue
        repo, number = row.get("repository"), row.get("pull_request")
        if type(number) is not int:
            match = re.fullmatch(r"([^#]+)#([0-9]+)", row.get("identity") or "")
            if match:
                repo, number = match.group(1), int(match.group(2))
        if type(repo) is str and type(number) is int:
            prior.add((repo, number))
    qualification_bytes = inputs.get(protocol["prior_qualification_path"])
    if qualification_bytes is None:
        raise ValueError("prior qualification catalog is not a hash-verified input")
    qualification = json.loads(qualification_bytes.decode("utf-8"))
    for task in qualification.get("tasks", []):
        if task.get("split") not in {"development", "train"}:
            continue
        repo = task.get("repository")
        task_id = task.get("task_id")
        match = re.search(r"-(\d+)(?:-|$)", task_id or "")
        if type(repo) is str and match:
            prior.add((repo, int(match.group(1))))
    for input_path in protocol["development_exposure_paths"]:
        ledger_bytes = inputs.get(input_path)
        if ledger_bytes is None:
            raise ValueError(f"development exposure is not hash verified: {input_path}")
        ledger = json.loads(ledger_bytes.decode("utf-8"))
        for event in ledger.get("events", []):
            repo, number = event.get("source_repository"), event.get("source_pull_request")
            if event.get("split") == "development" and type(repo) is str and type(number) is int:
                prior.add((repo, number))
    for row in protocol["additional_prior_identities"]:
        prior.add((row["repository"], row["number"]))
    return prior


def collect(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    inputs = frozen_inputs(protocol)
    prior = load_prior_identities(protocol, inputs)
    discovery = protocol["discovery"]
    searches: list[dict[str, Any]] = []
    search_rows: list[tuple[str, dict[str, Any]]] = []
    failures: list[dict[str, str]] = []
    for repo in discovery["repositories"]:
        try:
            total, rows = search_repository(
                repo, discovery["merged_date_start"], discovery["merged_date_end"],
            )
            scoped = [
                row for row in rows
                if type(row.get("mergedAt")) is str
                and row["mergedAt"] <= discovery["snapshot_cutoff_utc"]
            ]
            searches.append({
                "repository": repo,
                "search_result_count": total,
                "snapshot_scoped_count": len(scoped),
            })
            search_rows.extend((repo, row) for row in scoped)
        except Exception as exc:
            failures.append({"repository": repo, "error": f"{type(exc).__name__}: {exc}"})
    if len(search_rows) > discovery["maximum_pull_requests"]:
        failures.append({"repository": "multiple", "error": "metadata request ceiling exceeded"})

    rows: list[dict[str, Any]] = []
    if not failures:
        with ThreadPoolExecutor(max_workers=discovery["max_parallel_pull_queries"]) as pool:
            futures = {
                pool.submit(pull_metadata, repo, item["number"]): (repo, item)
                for repo, item in search_rows
            }
            for future in as_completed(futures):
                repo, search_item = futures[future]
                try:
                    row = future.result()
                    if row.get("number") != search_item.get("number"):
                        raise ValueError("search/detail pull-request numbers differ")
                    for key, search_key in (("title", "title"), ("url", "url"), ("state", "state"), ("merged_at", "mergedAt")):
                        if row.get(key) != search_item.get(search_key):
                            raise ValueError(f"search/detail pull-request {key} differs")
                    rows.append(row)
                except Exception as exc:
                    failures.append({
                        "repository": repo,
                        "number": str(search_item.get("number")),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    rows.sort(key=lambda row: (row["repository"], row["number"]))
    screen_rows = [screen_candidate(row, protocol=protocol, prior=prior) for row in rows]
    provisional = sum(row["coarse_screen"] == "provisional_for_manual_review" for row in screen_rows)
    output_dir.mkdir(parents=True)
    pool_record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GitHub GraphQL Search and PullRequest metadata fields only; no PR body, comments, patches, source, tests, or task assets fetched",
        "repositories": discovery["repositories"],
        "merged_date_window": [discovery["merged_date_start"], discovery["merged_date_end"]],
        "snapshot_cutoff_utc": discovery["snapshot_cutoff_utc"],
        "search_counts": searches,
        "candidate_count_captured": len(rows),
        "failures": failures,
        "candidates": rows,
    }
    pool_bytes = (json.dumps(pool_record, indent=2, sort_keys=True) + "\n").encode()
    pool_path = output_dir / "candidate-pool.json"
    pool_path.write_bytes(pool_bytes)
    screen_record = {
        "schema_version": 1,
        "screen_id": "mlsys-fresh-development-metadata-screen-v5",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_pool_sha256": sha256_bytes(pool_bytes),
        "candidate_count": len(screen_rows),
        "provisional_count": provisional,
        "excluded_count": len(screen_rows) - provisional,
        "failure_count": len(failures),
        "qualification_authorized": False,
        "development_assignment_authorized": False,
        "model_or_agent_inference_authorized": False,
        "heldout_access_authorized": False,
        "training_authorized": False,
        "candidates": screen_rows,
    }
    screen_bytes = (json.dumps(screen_record, indent=2, sort_keys=True) + "\n").encode()
    screen_path = output_dir / "screen.json"
    screen_path.write_bytes(screen_bytes)
    files = {
        "candidate-pool.json": {"bytes": len(pool_bytes), "sha256": sha256_bytes(pool_bytes)},
        "screen.json": {"bytes": len(screen_bytes), "sha256": sha256_bytes(screen_bytes)},
    }
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "collector_sha256": sha256_file(Path(__file__)),
        "prior_exposure_input_hashes": {
            input_path: sha256_bytes(payload) for input_path, payload in inputs.items()
        },
        "files": files,
    }
    (output_dir / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output_dir),
        "candidate_count": len(rows),
        "provisional_count": provisional,
        "excluded_count": len(rows) - provisional,
        "failure_count": len(failures),
        "candidate_pool_sha256": files["candidate-pool.json"]["sha256"],
        "screen_sha256": files["screen.json"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-discovery-v4":
        raise ValueError("unexpected discovery protocol ID")
    if sha256_file(Path(__file__)) != protocol.get("collector_sha256"):
        raise ValueError("collector source differs from the frozen protocol")
    result = collect(protocol, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
