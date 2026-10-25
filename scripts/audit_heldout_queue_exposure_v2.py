#!/usr/bin/env python3
"""Search retained VARE model/agent artifacts for three outside-pool candidate families."""

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

TASK_PATTERNS = {
    "transformers-49147-gemma3n-fp16-gradient-clipping": re.compile(
        rb"transformers[-_/ ]49147|#49147|test_from_pretrained_fp16_initializes_gradient_clipping|"
        rb"gemma3n.{0,120}gradient_clipping.{0,120}(fp16|float16)|"
        rb"(fp16|float16).{0,120}gradient_clipping.{0,120}gemma3n",
        re.IGNORECASE | re.DOTALL,
    ),
    "mlxlm-1925-minicpm3-phixtral-left-padding": re.compile(
        rb"mlxlm[-_/ ]1925|#1925|left_padding_test_runner|"
        rb"(minicpm3|phixtral).{0,120}left[- ]padding.{0,120}mask|"
        rb"left[- ]padding.{0,120}(minicpm3|phixtral)",
        re.IGNORECASE | re.DOTALL,
    ),
    "transformers-49116-qwen35-auto-tokenizer": re.compile(
        rb"transformers[-_/ ]49116|#49116|test_auto_skips_incorrect_hub_tokenizer_class|"
        rb"Qwen/Qwen3\.8-Flash-Next|Qwen/Qwen3\.6-27B|Qwen/Qwen3\.6-35B-A3B|"
        rb"Qwen/Qwen3\.8-2\.4T-A95B|qwen3_5_moe_text.{0,120}(tokenizer|AutoTokenizer)",
        re.IGNORECASE | re.DOTALL,
    ),
}

GIT_GREP_PATTERNS = [
    r"transformers[-_/ ]49147|#49147|test_from_pretrained_fp16_initializes_gradient_clipping|gemma3n.*gradient_clipping.*(fp16|float16)|(fp16|float16).*gradient_clipping.*gemma3n",
    r"mlxlm[-_/ ]1925|#1925|left_padding_test_runner|(minicpm3|phixtral).*left[- ]padding.*mask|left[- ]padding.*(minicpm3|phixtral)",
    r"transformers[-_/ ]49116|#49116|test_auto_skips_incorrect_hub_tokenizer_class|Qwen/Qwen3\.8-Flash-Next|Qwen/Qwen3\.6-27B|Qwen/Qwen3\.6-35B-A3B|Qwen/Qwen3\.8-2\.4T-A95B|qwen3_5_moe_text.*(tokenizer|AutoTokenizer)",
]
ARTIFACT_GLOBS = [
    "results/**/rollouts/**", "results/**/trajectories/**", "results/**/transcripts/**",
    "results/**/prompts/**", "results/**/agent*/**", "results/**/sft/**",
    "results/**/training*/**", "results/**/examples/**", "results/**/datasets/**",
]
ARTIFACT_SEGMENTS = {
    "rollout", "rollouts", "trajectory", "trajectories", "transcript", "transcripts",
    "prompt", "prompts", "agent", "agents", "training", "train", "sft", "dataset", "examples",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".md", ".csv", ".yaml", ".yml"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True, timeout=60,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    ).strip()


def is_artifact_path(path: str) -> bool:
    segments = [part.lower() for part in Path(path).parts]
    return "results" in segments and any(
        part in ARTIFACT_SEGMENTS
        or any(token in part for token in ("rollout", "trajectory", "transcript", "prompt", "agent", "training", "sft"))
        for part in segments
    )


def local_worktrees() -> list[Path]:
    roots = []
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            path = Path(line.removeprefix("worktree "))
            if (path / ".git").exists() and path not in roots:
                roots.append(path)
    return roots


def run_audit() -> dict:
    started = time.monotonic()
    refs = [line.split(" ", 1)[0] for line in git("for-each-ref", "--format=%(refname) %(objectname)").splitlines() if " " in line]
    git_matches = []
    git_paths_scanned = 0
    for ref in refs:
        listed = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-rz", "--name-only", "-r", ref, "--", *ARTIFACT_GLOBS],
            timeout=20, env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        git_paths_scanned += sum(bool(item) for item in listed.split(b"\0"))
        command = ["git", "-C", str(ROOT), "grep", "-I", "-i", "-n", "-E"]
        for pattern in GIT_GREP_PATTERNS:
            command.extend(("-e", pattern))
        command.extend((ref, "--", *ARTIFACT_GLOBS))
        found = subprocess.run(
            command, capture_output=True, timeout=30,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        if found.returncode not in (0, 1):
            raise RuntimeError(f"git grep failed on {ref}: {found.stderr.decode('utf-8', 'replace')[-400:]}")
        for row in found.stdout.splitlines():
            parts = row.split(b":", 3)
            if len(parts) < 4:
                continue
            path = parts[1].decode("utf-8", "replace")
            matched_line = parts[3]
            task_ids = [task_id for task_id, pattern in TASK_PATTERNS.items() if pattern.search(matched_line)]
            if not task_ids:
                continue
            blob = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{ref}:{path}"], timeout=30,
                env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
            )
            git_matches.append({
                "ref": ref, "path": path, "task_ids": task_ids,
                "content_sha256": sha(blob), "matched_line_sha256": sha(matched_line),
            })

    roots = local_worktrees()
    local_scanned = 0
    local_matches = {}
    large_skipped = []
    for root in roots:
        result_root = root / "results"
        if not result_root.is_dir():
            continue
        for path in result_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if not is_artifact_path(relative):
                continue
            try:
                size = path.stat().st_size
                if size > 50_000_000:
                    large_skipped.append({"worktree": str(root), "path": relative, "size_bytes": size})
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            local_scanned += 1
            task_ids = [task_id for task_id, pattern in TASK_PATTERNS.items() if pattern.search(raw)]
            if task_ids:
                local_matches[(str(root), relative)] = {
                    "worktree": str(root), "path": relative, "task_ids": task_ids,
                    "content_sha256": sha(raw),
                }

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))
    matches = git_matches + list(local_matches.values())
    return {
        "audit_id": "mlsys-heldout-queue-exposure-v2",
        "repository_head": git("rev-parse", "HEAD"),
        "implementation_sha256": sha(Path(__file__).read_bytes()),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_ids": list(TASK_PATTERNS),
        "retained_git_refs_scanned": len(refs),
        "artifact_paths_scanned_across_refs": git_paths_scanned,
        "local_worktrees": [str(root) for root in roots],
        "local_artifact_files_scanned": local_scanned,
        "local_large_files_skipped": large_skipped,
        "candidate_rollout_or_training_matches": sorted(matches, key=lambda row: (row.get("ref", ""), row["path"], row.get("worktree", ""))),
        "candidate_model_prompts": 0,
        "candidate_model_rollouts_started": False,
        "training_or_update_started": False,
        "external_pretraining_corpus": "UNKNOWN_NOT_AUDITED",
        "scope_limit": "Search covers retained VARE Git refs and local worktrees in the listed model/agent evidence paths only; it cannot establish absence from external pretraining, unmounted paths, or semantic paraphrases without the searched identifiers/symptoms.",
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "peak_rss_bytes": peak_rss,
        },
        "decision": "CLEAN_WITHIN_AUDITED_VARE_SCOPE" if not matches else "RETIRE_MATCHED_TASKS_PENDING_ADJUDICATION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("output exists; audit is append-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive_atomic(args.output, raw)
    print(json.dumps({
        "retained_git_refs_scanned": result["retained_git_refs_scanned"],
        "artifact_paths_scanned_across_refs": result["artifact_paths_scanned_across_refs"],
        "local_artifact_files_scanned": result["local_artifact_files_scanned"],
        "matches": len(result["candidate_rollout_or_training_matches"]),
        "decision": result["decision"],
    }, sort_keys=True))
    return 0 if result["decision"] == "CLEAN_WITHIN_AUDITED_VARE_SCOPE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
