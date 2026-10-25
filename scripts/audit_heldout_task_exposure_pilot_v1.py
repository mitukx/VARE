#!/usr/bin/env python3
"""Search retained VARE model/agent records for frozen task families."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.mlsys_heldout_task_pilot_v1 import write_exclusive_atomic  # noqa: E402

PATTERNS = {
    "mlxlm-1902-continuation-score-truncation": re.compile(
        rb"mlxlm[-_/ ]1902|#1902|scores_only_continuation_after_context_truncation|"
        rb"continuation.{0,48}(truncat|score)|truncat.{0,48}continuation",
        re.IGNORECASE | re.DOTALL,
    ),
    "mlxlm-1926-falcon-cache-gradient": re.compile(
        rb"mlxlm[-_/ ]1926|#1926|falcon_h1_without_cache|"
        rb"falcon[-_ ]?h1.{0,64}(cache|gradient)|"
        rb"(cache|gradient).{0,64}falcon[-_ ]?h1",
        re.IGNORECASE | re.DOTALL,
    ),
    "mlxlm-1938-phi3small-logits-mask": re.compile(
        rb"mlxlm[-_/ ]1938|#1938|test_phi3small|phi3small.{0,80}(dummy|logit|mask)|"
        rb"(dummy|logit|mask).{0,80}phi3small",
        re.IGNORECASE | re.DOTALL,
    ),
    "mlxlm-1979-ssm-padding-mask": re.compile(
        rb"mlxlm[-_/ ]1979|#1979|ssm_right_pad|"
        rb"(?<![a-z0-9_])ssm(?![a-z0-9_]).{0,40}(right[- ]?pad|padding)|"
        rb"(right[- ]?pad|padding).{0,40}(?<![a-z0-9_])ssm(?![a-z0-9_])|"
        rb"state[-_ ]space.{0,80}(right[- ]?pad|padding)|"
        rb"(right[- ]?pad|padding).{0,80}state[-_ ]space",
        re.IGNORECASE | re.DOTALL,
    ),
}
EXPOSURE_PATH_SEGMENTS = {
    "rollout", "rollouts", "trajectory", "trajectories", "transcript", "transcripts",
    "prompt", "prompts", "agent", "agents", "training", "train", "sft", "dataset", "examples",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".csv", ".yaml", ".yml"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        timeout=60,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    ).strip()


def _is_exposure_path(path: str) -> bool:
    segments = [segment.lower() for segment in Path(path).parts]
    return "results" in segments and any(
        segment in EXPOSURE_PATH_SEGMENTS or any(token in segment for token in ("rollout", "trajectory", "transcript", "prompt", "agent", "training", "sft"))
        for segment in segments
    )


def _worktree_roots() -> list[Path]:
    text = _git("worktree", "list", "--porcelain")
    roots = []
    for line in text.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree "):])
            if (path / ".git").exists() and path not in roots:
                roots.append(path)
    return roots


def audit() -> dict:
    started = time.monotonic()
    ref_rows = _git("for-each-ref", "--format=%(refname) %(objectname)").splitlines()
    refs = [(line.split(" ", 1)[0], line.split(" ", 1)[1]) for line in ref_rows if " " in line]
    paths = [
        "results/**/rollouts/**", "results/**/trajectories/**", "results/**/transcripts/**",
        "results/**/prompts/**", "results/**/agent*/**", "results/**/sft/**",
        "results/**/training*/**", "results/**/examples/**", "results/**/datasets/**",
    ]
    matches: list[dict] = []
    ref_scanned_files = 0
    for ref, _object_id in refs:
        listed = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-rz", "--name-only", "-r", ref, "--", *paths],
            timeout=15,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        ref_scanned_files += sum(bool(path) for path in listed.split(b"\0"))
        command = ["git", "-C", str(ROOT), "grep", "-I", "-i", "-n", "-E", "-e", "mlxlm[-_/ ]1902|#1902|scores_only_continuation_after_context_truncation|continuation.{0,48}(truncat|score)|truncat.{0,48}continuation|mlxlm[-_/ ]1926|#1926|falcon_h1_without_cache|falcon[-_ ]?h1.{0,64}(cache|gradient)|(cache|gradient).{0,64}falcon[-_ ]?h1|mlxlm[-_/ ]1938|#1938|test_phi3small|phi3small.{0,80}(dummy|logit|mask)|(dummy|logit|mask).{0,80}phi3small|mlxlm[-_/ ]1979|#1979|ssm_right_pad|(^|[^[:alnum:]_])ssm([^[:alnum:]_]|$).{0,40}(right[- ]?pad|padding)|(right[- ]?pad|padding).{0,40}(^|[^[:alnum:]_])ssm([^[:alnum:]_]|$)|state[-_ ]space.{0,80}(right[- ]?pad|padding)|(right[- ]?pad|padding).{0,80}state[-_ ]space", ref, "--", *paths]
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=15,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(f"git grep failed at {ref}: {result.stderr.decode('utf-8','replace')[-500:]}")
        output = result.stdout.decode("utf-8", "replace")
        found_paths = set()
        for line in output.splitlines():
            # Git grep format is ref:path:line:matched-text.
            parts = line.split(":", 3)
            if len(parts) < 3:
                continue
            path = parts[1]
            found_paths.add(path)
            matches.append({
                "ref": ref,
                "path": path,
                "line": parts[2],
                "match_class": "retained_model_or_agent_artifact",
                "content_sha256": sha(line.encode("utf-8", "replace")),
            })

    local_files = 0
    local_skipped_large = []
    local_matches: dict[tuple[str, str], dict] = {}
    roots = _worktree_roots()
    for root in roots:
        results_root = root / "results"
        if not results_root.is_dir():
            continue
        for path in results_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if not _is_exposure_path(relative):
                continue
            try:
                if path.stat().st_size > 50_000_000:
                    local_skipped_large.append({"worktree": str(root), "path": relative, "size_bytes": path.stat().st_size})
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            local_files += 1
            for task_id, pattern in PATTERNS.items():
                if pattern.search(raw):
                    local_matches[(str(root), relative)] = {
                        "worktree": str(root),
                        "path": relative,
                        "content_sha256": sha(raw),
                        "task_ids": [task_id],
                        "match_class": "local_model_or_agent_artifact",
                    }

    elapsed = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    artifact_matches = sorted(matches + list(local_matches.values()), key=lambda x: (x.get("ref", ""), x["path"], x.get("worktree", "")))
    return {
        "audit_id": "mlsys-heldout-task-exposure-pilot-v1",
        "repository_head": _git("rev-parse", "HEAD"),
        "audit_implementation_sha256": sha(Path(__file__).resolve().read_bytes()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_ids": list(PATTERNS),
        "retained_git_refs": len(refs),
        "git_exposure_candidate_paths_scanned_across_refs": ref_scanned_files,
        "local_worktrees": [str(root) for root in roots],
        "local_exposure_files_scanned": local_files,
        "local_large_files_skipped": local_skipped_large,
        "candidate_rollout_or_training_matches": artifact_matches,
        "candidate_model_prompts": 0,
        "candidate_model_rollouts_started": False,
        "training_or_update_started": False,
        "observed_vare_candidate_exposure": bool(artifact_matches),
        "external_pretraining_corpus": "UNKNOWN_NOT_AUDITED",
        "scope_limit": "Search covers the listed retained VARE refs/worktrees and identified model/agent-artifact paths only. It cannot prove absence from files outside this inventory, public model pretraining corpora, or semantically paraphrased records that contain none of the frozen identifiers/symptoms.",
        "resources": {
            "wall_seconds": elapsed,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "peak_rss_bytes": peak_rss,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("output exists; exposure audit is append-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = audit()
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive_atomic(args.output, raw)
    print(json.dumps({
        "retained_git_refs": result["retained_git_refs"],
        "git_exposure_candidate_paths_scanned_across_refs": result["git_exposure_candidate_paths_scanned_across_refs"],
        "local_exposure_files_scanned": result["local_exposure_files_scanned"],
        "candidate_exposure_matches": len(result["candidate_rollout_or_training_matches"]),
        "decision": "CLEAN_WITHIN_AUDITED_VARE_SCOPE" if not result["observed_vare_candidate_exposure"] else "EXPOSED_TASK_RETIRED_FROM_HELDOUT",
    }, sort_keys=True))
    return 0 if not result["observed_vare_candidate_exposure"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
