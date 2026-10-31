#!/usr/bin/env python3
"""Capture a metadata-only, post-cutoff candidate pool for later task review.

This script intentionally does not fetch source diffs, tests, archives, or task
patches. It records GitHub PR metadata and changed paths only. A provisional
screen is not task qualification and does not authorize model or agent use.
"""
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
PROTOCOL_PATH = ROOT / "protocols/mlsys_fresh_development_candidate_discovery_v1.json"
QUEUE_AUDIT_PATH = ROOT / "results/mlsys-candidate-manual-qualification-queue-exposure-v1.json"
QUALIFICATION_PATH = ROOT / "results/mlsys-task-deep-qualification-v1/qualification.json"
GITHUB_PR_SEARCH_QUERY = """
query($query: String!, $cursor: String) {
  search(query: $query, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest { number title url state isDraft createdAt updatedAt mergedAt }
    }
  }
}
"""
GITHUB_GRAPHQL_QUERY = """
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
SPECIALIZED_TITLE_MARKERS = (
    "gpu", "metal", "cuda", "fsdp", "distributed", "expert parallel",
    "neuron", "architecture", "support ", "add ", "integration test",
    "preemption",
)
TEST_PATH_MARKERS = ("test", "spec", "fixture")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "mlsys-fresh-development-candidate-discovery-v1":
        raise ValueError("unexpected discovery protocol ID")
    discovery = protocol.get("discovery")
    if type(discovery) is not dict:
        raise ValueError("discovery protocol section is missing")
    if discovery.get("collector_sha256") != sha256_file(Path(__file__)):
        raise ValueError("collector source differs from the frozen protocol")
    if discovery.get("queue_audit_sha256") != sha256_file(QUEUE_AUDIT_PATH):
        raise ValueError("prior-exposure audit differs from the frozen protocol")
    if discovery.get("prior_qualification_catalog_sha256") != sha256_file(QUALIFICATION_PATH):
        raise ValueError("prior qualification catalog differs from the frozen protocol")
    return protocol


def run_gh_json(arguments: list[str], *, timeout: int = 90) -> Any:
    result = subprocess.run(
        ["gh", "api", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return json.loads(result.stdout)


def run_gh_graphql(query: str, variables: dict[str, str | int]) -> dict[str, Any]:
    arguments = ["graphql", "-f", f"query={query}"]
    for name, value in variables.items():
        flag = "-F" if type(value) is int else "-f"
        arguments.extend([flag, f"{name}={value}"])
    payload = run_gh_json(arguments)
    if type(payload) is not dict or payload.get("errors"):
        raise ValueError(f"GitHub GraphQL query failed: {(payload or {}).get('errors')}")
    return payload


def search_repo_month(repo: str, date_start: str, date_end: str) -> list[dict[str, Any]]:
    search_text = f"repo:{repo} is:pr is:merged merged:{date_start}..{date_end}"
    page = run_gh_graphql(GITHUB_PR_SEARCH_QUERY, {"query": search_text})["data"]["search"]
    total = page.get("issueCount")
    if type(total) is not int:
        raise ValueError(f"incomplete GitHub GraphQL search response for {repo} {date_start}..{date_end}")
    if total > 1000:
        raise ValueError(f"GitHub Search 1,000-result ceiling exceeded for {repo} {date_start}..{date_end}")
    nodes = page.get("nodes")
    info = page.get("pageInfo")
    if type(nodes) is not list or type(info) is not dict:
        raise ValueError(f"GitHub search response has no nodes/pageInfo for {repo} {date_start}..{date_end}")
    rows = list(nodes)
    cursor = info.get("endCursor")
    while info.get("hasNextPage"):
        if type(cursor) is not str:
            raise ValueError(f"GitHub search cursor missing for {repo} {date_start}..{date_end}")
        next_page = run_gh_graphql(
            GITHUB_PR_SEARCH_QUERY,
            {"query": search_text, "cursor": cursor},
        )["data"]["search"]
        next_nodes = next_page.get("nodes")
        info = next_page.get("pageInfo")
        if type(next_nodes) is not list or type(info) is not dict or not next_nodes:
            raise ValueError(f"GitHub search ended before total_count for {repo} {date_start}..{date_end}")
        rows.extend(next_nodes)
        cursor = info.get("endCursor")
    if len(rows) != total:
        raise ValueError(f"GitHub search item count differs from total_count for {repo} {date_start}..{date_end}")
    if any(type(item) is not dict or type(item.get("number")) is not int for item in rows):
        raise ValueError(f"GitHub returned a non-pull-request node for {repo} {date_start}..{date_end}")
    return rows


def pull_metadata(repo: str, number: int) -> dict[str, Any]:
    owner, name = repo.split("/", maxsplit=1)
    payload = run_gh_graphql(GITHUB_GRAPHQL_QUERY, {"owner": owner, "name": name, "number": number})
    pr = ((payload.get("data") or {}).get("repository") or {}).get("pullRequest")
    if type(pr) is not dict:
        raise ValueError(f"GitHub returned no pull request for {repo}#{number}")
    files = pr.get("files")
    nodes = files.get("nodes") if type(files) is dict else None
    if type(nodes) is not list or any(type(row) is not dict or type(row.get("path")) is not str for row in nodes):
        raise ValueError(f"GitHub returned malformed changed paths for {repo}#{number}")
    merge_commit = pr.get("mergeCommit")
    if merge_commit is not None and type(merge_commit) is not dict:
        raise ValueError(f"GitHub returned malformed merge commit for {repo}#{number}")
    merge_parents = (merge_commit or {}).get("parents", {}).get("nodes", [])
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
        "merge_base_sha": merge_parents[0].get("oid") if merge_parents else None,
        "head_sha": pr.get("headRefOid"),
        "merge_commit_sha": merge_commit.get("oid") if merge_commit else None,
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_file_count": pr.get("changedFiles"),
        "changed_paths": [row["path"] for row in nodes],
        "changed_paths_complete": not files["pageInfo"].get("hasNextPage", False),
        "search_result_number": number,
    }


def is_test_path(path: str) -> bool:
    lower = path.lower()
    return any(marker in lower for marker in TEST_PATH_MARKERS)


def screen_candidate(
    row: dict[str, Any],
    *,
    date_start: str,
    date_end: str,
    cutoff_proxy: str,
    prior_identities: set[tuple[str, int]],
) -> dict[str, Any]:
    reasons: list[str] = []
    number = row.get("number")
    identity = (row.get("repository"), number) if type(number) is int else None
    merged_date = (row.get("merged_at") or "")[:10]
    if (
        row.get("state") != "CLOSED"
        or not row.get("merged_at")
        or not row.get("merge_commit_sha")
        or not row.get("merge_base_sha")
    ):
        reasons.append("not_merged_or_missing_merge_commit")
    if not date_start <= merged_date <= date_end:
        reasons.append("merged_outside_frozen_date_window")
    if merged_date <= cutoff_proxy:
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
    elif additions + deletions > 250:
        reasons.append("diff_exceeds_250_changed_lines")
    changed_file_count = row.get("changed_file_count")
    if type(changed_file_count) is not int:
        reasons.append("changed_file_count_missing")
    elif changed_file_count > 8:
        reasons.append("diff_exceeds_8_files")
    title = (row.get("title") or "").lower()
    if any(marker in title for marker in SPECIALIZED_TITLE_MARKERS):
        reasons.append("title_flags_specialized_or_feature_scope_for_manual_review")
    if identity in prior_identities:
        reasons.append("exact_identity_already_in_vare_candidate_or_exposure_audit")
    return {
        **row,
        "test_paths": test_paths,
        "coarse_screen": "provisional_for_manual_review" if not reasons else "excluded_from_shortlist",
        "screen_reasons": reasons,
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
    }


def collect(protocol: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    discovery = protocol["discovery"]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    queue_audit = json.loads(QUEUE_AUDIT_PATH.read_text(encoding="utf-8"))
    prior_identities: set[tuple[str, int]] = set()
    for row in queue_audit.get("candidates", []):
        identity = row.get("identity")
        match = re.fullmatch(r"([^/]+/[^#]+)#([0-9]+)", identity or "")
        if match:
            prior_identities.add((match.group(1), int(match.group(2))))
    for row in discovery["additional_prior_identities"]:
        prior_identities.add((row["repository"], row["number"]))

    search_results: list[tuple[str, int, dict[str, Any]]] = []
    search_counts: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for repo in discovery["repositories"]:
        try:
            items = search_repo_month(repo, discovery["merged_date_start"], discovery["merged_date_end"])
            search_counts.append({"repository": repo, "search_result_count": len(items)})
            for item in items:
                number = item.get("number")
                if type(number) is not int:
                    raise ValueError(f"search result is not a pull request in {repo}")
                search_results.append((repo, number, item))
        except Exception as exc:
            failures.append({"repository": repo, "error": str(exc)})
    metadata_rows: list[dict[str, Any]] = []
    if not failures:
        with ThreadPoolExecutor(max_workers=discovery["max_parallel_pull_queries"]) as pool:
            futures = {pool.submit(pull_metadata, repo, number): (repo, number, item) for repo, number, item in search_results}
            for future in as_completed(futures):
                repo, number, search_item = futures[future]
                try:
                    row = future.result()
                    if row.get("number") != number:
                        raise ValueError("PR number differs from search result")
                    for key in ("title", "url", "merged_at"):
                        if row.get(key) != search_item.get({"merged_at": "mergedAt"}.get(key, key)):
                            raise ValueError(f"search and pull metadata differ for {key}")
                    metadata_rows.append(row)
                except Exception as exc:
                    failures.append({"repository": repo, "number": str(number), "error": str(exc)})
    metadata_rows.sort(key=lambda row: (row["repository"], row["number"]))
    screen_rows = [screen_candidate(
        row,
        date_start=discovery["merged_date_start"],
        date_end=discovery["merged_date_end"],
        cutoff_proxy=discovery["model_training_cutoff_proxy"],
        prior_identities=prior_identities,
    ) for row in metadata_rows]
    pool_record = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GitHub Search Issues API and GitHub GraphQL pull-request metadata; no source diffs or test content fetched",
        "repositories": discovery["repositories"],
        "merged_date_start": discovery["merged_date_start"],
        "merged_date_end": discovery["merged_date_end"],
        "search_counts": search_counts,
        "candidate_count_requested_from_search": sum(row["search_result_count"] for row in search_counts),
        "candidate_count_captured": len(metadata_rows),
        "failures": failures,
        "candidates": metadata_rows,
    }
    pool_bytes = (json.dumps(pool_record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    pool_path = output_dir / "candidate-pool.json"
    pool_path.write_bytes(pool_bytes)
    screen_record = {
        "schema_version": 1,
        "screen_id": "mlsys-fresh-development-metadata-screen-v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_pool_sha256": sha256_bytes(pool_bytes),
        "model_training_cutoff_proxy": discovery["model_training_cutoff_proxy"],
        "candidate_count": len(screen_rows),
        "provisional_count": sum(row["coarse_screen"] == "provisional_for_manual_review" for row in screen_rows),
        "excluded_count": sum(row["coarse_screen"] == "excluded_from_shortlist" for row in screen_rows),
        "qualification_authorized": False,
        "model_or_agent_inference_authorized": False,
        "training_authorized": False,
        "rule": {
            "merged_date_window": [discovery["merged_date_start"], discovery["merged_date_end"]],
            "strictly_after_model_training_cutoff_proxy": discovery["model_training_cutoff_proxy"],
            "test_path_in_changed_file_metadata_required": True,
            "maximum_changed_lines": 250,
            "maximum_changed_files": 8,
            "specialized_title_markers": list(SPECIALIZED_TITLE_MARKERS),
            "exclude_exact_prior_candidate_or_exposure_identity": True,
            "meaning": "metadata-only triage; not task qualification, split assignment, or proof of pretraining freshness",
        },
        "candidates": screen_rows,
    }
    screen_bytes = (json.dumps(screen_record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    screen_path = output_dir / "screen.json"
    screen_path.write_bytes(screen_bytes)
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "collector_sha256": sha256_file(Path(__file__)),
        "queue_audit_sha256": sha256_file(QUEUE_AUDIT_PATH),
        "files": {
            "candidate-pool.json": {"bytes": len(pool_bytes), "sha256": sha256_bytes(pool_bytes)},
            "screen.json": {"bytes": len(screen_bytes), "sha256": sha256_bytes(screen_bytes)},
        },
    }
    manifest_path = output_dir / "evidence-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output": str(output_dir),
        "candidate_count": len(metadata_rows),
        "provisional_count": screen_record["provisional_count"],
        "excluded_count": screen_record["excluded_count"],
        "failure_count": len(failures),
        "candidate_pool_sha256": manifest["files"]["candidate-pool.json"]["sha256"],
        "screen_sha256": manifest["files"]["screen.json"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    result = collect(protocol, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["failure_count"] == 0 and result["candidate_count"] == sum(
        row["search_result_count"] for row in json.loads(
            (Path(result["output"]) / "candidate-pool.json").read_text(encoding="utf-8")
        )["search_counts"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
