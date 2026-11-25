#!/usr/bin/env python3
"""Prepare a shallow, offline, single-file candidate checkout for the task."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess


BASELINE = "b47ec788c4ea3051b207afd94281fb75594b0205"
KNOWN_FIX = "676859711989dd9178fbbe8968e5a1f86d519712"
TARGET_PATH = "src/transformers/integrations/mxfp4.py"
TARGET_SHA256 = "ec83a032b49ae7bbc7dce3de04b45dab2b715187e84b0806eeb3a73738780bb5"
REPOSITORY = "https://github.com/huggingface/transformers.git"


def git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout.strip()


def prepare(output: Path) -> dict[str, str]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite workspace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(output)], check=True, timeout=30)
    git_dir = output / ".git"
    (git_dir / "info/sparse-checkout").write_text(
        f"/{TARGET_PATH}\n", encoding="utf-8"
    )
    git(output, "config", "core.sparseCheckout", "true")
    git(output, "remote", "add", "origin", REPOSITORY)
    git(
        output,
        "-c",
        "protocol.version=2",
        "fetch",
        "--depth=1",
        "--filter=blob:none",
        "--no-tags",
        "origin",
        BASELINE,
    )
    git(output, "checkout", "--detach", "FETCH_HEAD")
    git(output, "remote", "remove", "origin")
    try:
        git(output, "config", "--unset", "extensions.partialClone")
    except subprocess.CalledProcessError:
        pass
    head = git(output, "rev-parse", "HEAD")
    shallow = git(output, "rev-parse", "--is-shallow-repository")
    remotes = git(output, "remote")
    source_sha256 = hashlib.sha256((output / TARGET_PATH).read_bytes()).hexdigest()
    try:
        git(output, "cat-file", "-e", f"{KNOWN_FIX}^{{commit}}")
        known_fix_present = "true"
    except subprocess.CalledProcessError:
        known_fix_present = "false"
    if (
        head != BASELINE
        or shallow != "true"
        or remotes
        or source_sha256 != TARGET_SHA256
        or known_fix_present != "false"
    ):
        raise RuntimeError("prepared checkout failed its baseline, source, or exposure checks")
    return {
        "head": head,
        "shallow": shallow,
        "remotes": remotes,
        "editable_source_sha256": source_sha256,
        "known_fix_commit_present": known_fix_present,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
