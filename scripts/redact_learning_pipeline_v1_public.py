#!/usr/bin/env python3
"""Regenerate and verify redacted public copies of learning-pipeline v1 evidence.

Some committed learning-pipeline v1 files record absolute paths of the hosts that produced them.
Those originals are hash-bound evidence and stay unchanged. This script derives public copies
under ``results/learning-pipeline-v1/public-redacted/<original repository-relative path>`` by
applying a fixed, ordered list of path substitutions. It then checks that:

* every original still has the sha256 recorded in ``MAPPING.json`` (and in ``SHA256SUMS.txt`` where
  the run lists it);
* every JSON / JSON Lines copy is valid and parses to the same structure as the original once the
  same substitutions are applied to its strings;
* no absolute host path or local directory label remains in any copy;
* no other file in the learning-pipeline v1 trees contains a host path without having a copy.

Usage:
  python scripts/redact_learning_pipeline_v1_public.py --write   # regenerate copies and MAPPING.json
  python scripts/redact_learning_pipeline_v1_public.py --check   # verify (exit 1 on any problem)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEST = Path("results/learning-pipeline-v1/public-redacted")
MAPPING = DEST / "MAPPING.json"
SCHEMA = "vare_public_redaction_v1"
SCOPE = (Path("results/learning-pipeline-v1"), Path("configs/learning_pipeline_v1"))
RUN_SUMS = Path("results/learning-pipeline-v1/run-001/SHA256SUMS.txt")

# Ordered regular-expression substitutions; the more specific run-directory rule comes first.
# Path components are matched generically so that no host-specific directory name is spelled here.
_SEG = r"[^/\s\"\\\x1b]+"
RULES: tuple[tuple[str, str], ...] = (
    (rf"/Users/{_SEG}/{_SEG}/artifacts/learning_pipeline_v1/run-001", "<RUN_ROOT>"),
    (rf"/Users/{_SEG}/[^/\s\"\\\x1b.]+(?=/(?:artifacts|protocols)/)", "<REPO_ROOT>"),
    (rf"/Users/{_SEG}/\.cache/huggingface/hub", "<HF_HUB_CACHE>"),
    (rf"/workspace/{_SEG}/lp-cache", "<DATA_CACHE>"),
)
PLACEHOLDERS = {
    "<RUN_ROOT>": "run-001 output directory on the training host",
    "<REPO_ROOT>": "repository checkout on the training host",
    "<HF_HUB_CACHE>": "Hugging Face hub cache directory on the training host",
    "<DATA_CACHE>": "dataset cache directory on the data-preparation host",
}

# Anything that looks like an absolute host path or a local directory label.
HOST_PATH = re.compile(
    r"(?:/Users/|/home/|/private/|/var/folders/|/Volumes/|/root/|/workspace/|/opt/homebrew/|/mnt/"
    r"|[A-Za-z]:\\\\Users\\\\|[A-Za-z]:\\Users\\)"
)

SOURCES: tuple[str, ...] = (
    "configs/learning_pipeline_v1/build_report.json",
    "results/learning-pipeline-v1/run-001/adapters/rft_self/TRAINING_COMPLETE.json",
    "results/learning-pipeline-v1/run-001/adapters/rft_self/adapter_config.json",
    "results/learning-pipeline-v1/run-001/adapters/rft_self/lora_config.json",
    "results/learning-pipeline-v1/run-001/adapters/sft_reference/TRAINING_COMPLETE.json",
    "results/learning-pipeline-v1/run-001/adapters/sft_reference/adapter_config.json",
    "results/learning-pipeline-v1/run-001/adapters/sft_reference/lora_config.json",
    "results/learning-pipeline-v1/run-001/eval/no_update/graded.jsonl",
    "results/learning-pipeline-v1/run-001/eval/rft_self/graded.jsonl",
    "results/learning-pipeline-v1/run-001/eval/sft_reference/graded.jsonl",
    "results/learning-pipeline-v1/run-001/events.jsonl",
    "results/learning-pipeline-v1/run-001/preflight_train_smoke/train.log",
    "results/learning-pipeline-v1/run-001/run_meta.json",
    "results/learning-pipeline-v1/run-001/train_report.json",
    "results/learning-pipeline-v1/run-001/train_rft_self.log",
    "results/learning-pipeline-v1/run-001/train_sft_reference.log",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact_text(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for pattern, placeholder in RULES:
        text, n = re.subn(pattern, placeholder, text)
        if n:
            counts[placeholder] = counts.get(placeholder, 0) + n
    return text, counts


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, dict):
        return {redact_text(k)[0]: redact_value(v) for k, v in value.items()}
    return value


def parse(path: str, text: str) -> list[Any]:
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)]


def host_path_hits(text: str) -> list[str]:
    return sorted({m.group(0) for m in HOST_PATH.finditer(text)})


def build(root: Path = ROOT) -> tuple[dict[str, Any], dict[Path, bytes], list[str]]:
    problems: list[str] = []
    copies: dict[Path, bytes] = {}
    files = []
    for rel in SOURCES:
        raw = (root / rel).read_bytes()
        text = raw.decode("utf-8")
        red, counts = redact_text(text)
        out_rel = DEST / rel
        if not counts:
            problems.append(f"{rel}: no rule matched")
        hits = host_path_hits(red)
        if hits:
            problems.append(f"{rel}: host path remains after redaction: {hits}")
        try:
            if rel.endswith((".json", ".jsonl")) and parse(rel, red) != [redact_value(v) for v in parse(rel, text)]:
                problems.append(f"{rel}: redacted copy is not the original structure with redacted strings")
        except json.JSONDecodeError as exc:
            problems.append(f"{rel}: redacted copy is not valid JSON: {exc}")
        data = red.encode("utf-8")
        copies[out_rel] = data
        files.append({"original": rel, "original_sha256": sha256(raw), "original_bytes": len(raw),
                      "redacted": out_rel.as_posix(), "redacted_sha256": sha256(data),
                      "redacted_bytes": len(data), "replacements": counts})
    mapping = {
        "schema": SCHEMA,
        "rule": {"type": "ordered regular-expression substitution (Python re) on the UTF-8 file text; "
                         "each rule is applied to the output of the previous one",
                 "substitutions": [{"pattern": p, "placeholder": ph} for p, ph in RULES],
                 "placeholders": PLACEHOLDERS,
                 "verification_pattern": HOST_PATH.pattern},
        "files": files,
    }
    return mapping, copies, problems


def scope_problems(root: Path = ROOT) -> list[str]:
    """Every file in the learning-pipeline v1 trees with a host path must have a copy."""
    problems = []
    listed = set(SOURCES)
    for base in SCOPE:
        for p in sorted((root / base).rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or rel.startswith(DEST.as_posix() + "/") or rel in listed:
                continue
            if p.name == Path(__file__).name:
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            red = redact_text(text)[0]
            if host_path_hits(red) or red != text:
                problems.append(f"{rel}: contains a host path but has no redacted copy")
    return problems


def original_hash_problems(mapping: dict[str, Any], root: Path = ROOT) -> list[str]:
    problems = []
    sums_path = root / RUN_SUMS
    recorded: dict[str, str] = {}
    if sums_path.exists():
        for line in sums_path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                recorded[(RUN_SUMS.parent / parts[-1]).as_posix()] = parts[0]
    for f in mapping["files"]:
        want = recorded.get(f["original"])
        if want and want != f["original_sha256"]:
            problems.append(f"{f['original']}: sha256 differs from {RUN_SUMS.as_posix()}")
    return problems


def check(root: Path = ROOT) -> list[str]:
    mapping, copies, problems = build(root)
    problems += scope_problems(root) + original_hash_problems(mapping, root)
    committed_path = root / MAPPING
    if not committed_path.exists():
        return [*problems, f"{MAPPING.as_posix()} missing"]
    committed = json.loads(committed_path.read_text())
    if committed != mapping:
        problems.append(f"{MAPPING.as_posix()} differs from the regenerated mapping "
                        "(an original changed or the rule changed)")
    for rel, data in copies.items():
        p = root / rel
        if not p.exists() or p.read_bytes() != data:
            problems.append(f"{rel.as_posix()}: committed copy differs from the regenerated copy")
    for p in sorted((root / DEST).rglob("*")):
        if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".log"}:
            text = p.read_text(encoding="utf-8")
            if p == root / MAPPING:  # the rule block holds the patterns themselves, not host paths
                text = json.dumps({k: v for k, v in json.loads(text).items() if k != "rule"})
            hits = host_path_hits(text)
            if hits:
                problems.append(f"{p.relative_to(root).as_posix()}: host path present: {hits}")
    return problems


def write(root: Path = ROOT) -> list[str]:
    mapping, copies, problems = build(root)
    problems += scope_problems(root) + original_hash_problems(mapping, root)
    if problems:
        return problems
    for rel, data in copies.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (root / MAPPING).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    problems = write() if args.write else check()
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    if not problems:
        print(f"OK {len(SOURCES)} redacted copies verified")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
