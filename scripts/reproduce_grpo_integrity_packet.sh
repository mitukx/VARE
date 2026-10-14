#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 PYTHON_BIN RVL_SRC_SYSTEMS OUTPUT_DIR" >&2
  exit 2
fi

python_bin=$1
rvl_source=$2
output=$3
repo_root=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -x "$python_bin" ]; then
  echo "PYTHON_BIN must be an executable Python binary" >&2
  exit 2
fi
for file in backends.py grpo.py hf_trainer.py triton_grpo.py types.py; do
  if [ ! -f "$rvl_source/$file" ]; then
    echo "RVL source file missing: $rvl_source/$file" >&2
    exit 2
  fi
done
if [ -e "$output" ]; then
  echo "Refusing to overwrite existing output: $output" >&2
  exit 2
fi
mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(cd "$output" && pwd)

on_exit() {
  status=$?
  if [ "$status" -ne 0 ]; then
    "$python_bin" - "$output" "$status" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], "failure.json").write_text(
    json.dumps({"status": "failed", "exit_code": int(sys.argv[2])}, indent=2) + "\n",
    encoding="utf-8")
PY
  fi
}
trap on_exit EXIT

cd "$repo_root"
run_logged() {
  name=$1
  shift
  echo "== $name =="
  "$@" 2>&1 | tee "$output/$name.log"
}

run_logged group_integrity_tests env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$python_bin" -m pytest -q \
  tests/test_grouped_replay.py \
  tests/test_replay_group_freshness.py \
  tests/test_rvl_grpo_hooks.py
run_logged candidate_boundary_rollback "$python_bin" \
  scripts/validate_rvl_grpo_partial_failure.py --rvl-source "$rvl_source"
run_logged in_step_optimizer_rollback "$python_bin" \
  scripts/validate_rvl_grpo_midstep_fault.py --rvl-source "$rvl_source"

"$python_bin" - "$repo_root" "$rvl_source" "$output" <<'PY'
import hashlib, importlib.metadata, json, platform, subprocess, sys
from pathlib import Path
repo_root, rvl_source, output = map(Path, sys.argv[1:])

def git_value(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()

source_hashes = {}
for name in ("backends.py", "grpo.py", "hf_trainer.py", "triton_grpo.py", "types.py"):
    source_hashes[name] = hashlib.sha256((rvl_source / name).read_bytes()).hexdigest()

metadata = {
    "schema_version": 1,
    "vare_revision": git_value(repo_root, "rev-parse", "HEAD"),
    "vare_worktree_clean": not bool(git_value(repo_root, "status", "--porcelain")),
    "rvl_revision": git_value(rvl_source, "rev-parse", "HEAD"),
    "rvl_worktree_clean": not bool(git_value(rvl_source, "status", "--porcelain")),
    "rvl_source_sha256": source_hashes,
    "runtime": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
    },
}
(output / "reproduction.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

files = {}
for path in sorted(output.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
(output / "manifest.json").write_text(
    json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
PY
trap - EXIT
echo "Evidence packet written to $output"
