#!/usr/bin/env python3
"""Collect a frozen discovery pool of upstream coding-fix PR metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "huggingface/transformers": [
        49412, 49208, 49123, 49147, 49116, 48744, 48319, 48205, 47340,
        45818, 45852, 48444, 49500, 48198, 43226,
    ],
    "ml-explore/mlx-lm": [
        1930, 1926, 1856, 1787, 1843, 1795, 1452, 986, 511, 288, 280,
        1938, 1925, 1921, 1912, 1910, 1904, 1902, 1898, 1935,
    ],
    "ml-explore/mlx": [
        4721, 4719, 4708, 4689, 4687, 4679, 4673, 4672, 4668, 4663, 4662,
    ],
    "huggingface/trl": [7768, 7748, 7719, 7723, 7724, 7731],
    "vllm-project/vllm": [61000, 56841, 55285, 52626],
    "ray-project/ray": [64360],
}


def gh_json(endpoint: str) -> object:
    result = subprocess.run(
        ["gh", "api", endpoint], check=True, capture_output=True, text=True, timeout=45,
    )
    return json.loads(result.stdout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_one(repo: str, number: int) -> dict:
    pr = gh_json(f"repos/{repo}/pulls/{number}")
    if not isinstance(pr, dict):
        raise TypeError(f"invalid PR payload: {repo}#{number}")
    files = gh_json(f"repos/{repo}/pulls/{number}/files?per_page=100")
    if not isinstance(files, list):
        raise TypeError(f"invalid file payload: {repo}#{number}")
    paths = [item.get("filename") for item in files if isinstance(item, dict)]
    return {
        "repository": repo,
        "number": number,
        "url": pr.get("html_url"),
        "title": pr.get("title"),
        "state": pr.get("state"),
        "merged": pr.get("merged"),
        "created_at": pr.get("created_at"),
        "merged_at": pr.get("merged_at"),
        "base_sha": (pr.get("base") or {}).get("sha"),
        "head_sha": (pr.get("head") or {}).get("sha"),
        "merge_commit_sha": pr.get("merge_commit_sha"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "changed_file_count": pr.get("changed_files"),
        "changed_paths": paths,
        "test_paths": [
            path for path in paths
            if path and any(token in path.lower() for token in ("test", "spec", "fixture"))
        ],
    }


def collect(output: Path) -> dict:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(collect_one, repo, number): (repo, number)
            for repo, numbers in CANDIDATES.items()
            for number in numbers
        }
        for future in as_completed(futures):
            repo, number = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # retain a visible inventory failure
                failures.append({"repository": repo, "number": number, "error": str(exc)})
    rows.sort(key=lambda row: (row["repository"], row["number"]))
    record = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "GitHub REST pull request and changed-file metadata",
        "search_date": "2026-10-20",
        "candidate_count_requested": sum(map(len, CANDIDATES.values())),
        "candidate_count_captured": len(rows),
        "failures": failures,
        "candidates": rows,
    }
    raw = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path = output / "candidate-pool.json"
    path.write_text(raw, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "files": {
            "candidate-pool.json": {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            },
        },
    }
    (output / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "output": str(output),
        "candidate_count_requested": record["candidate_count_requested"],
        "candidate_count_captured": len(rows),
        "failures": failures,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
