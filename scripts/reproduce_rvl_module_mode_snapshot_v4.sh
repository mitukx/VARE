#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-python3.12}
commit=c7e646b043cb56e5ea3c2623bb8a61e065451f72
if [[ $# -ne 1 ]]; then
  echo "usage: $0 EMPTY_OUTPUT_DIRECTORY" >&2
  exit 2
fi
out=$1
mkdir -p "$out"
if [[ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "output directory must be empty: $out" >&2
  exit 2
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"$python" -c 'import sys, torch, transformers; assert sys.version_info >= (3, 11); print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, Transformers {transformers.__version__}")' >"$out/runtime.txt"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

baseline="$tmp/baseline"
git clone --quiet --no-checkout https://github.com/mitukx/Recursive-Verification-Lag.git "$baseline"
git -C "$baseline" checkout --quiet --detach "$commit"
test "$(git -C "$baseline" rev-parse HEAD)" = "$commit"
git -C "$baseline" apply "$repo/results/rvl-module-mode-upstream-v4/test-only.patch"
set +e
(cd "$baseline" && "$python" -m unittest \
  tests.test_mini_lab_torch.TorchAcceptanceTests.test_hf_trainer_transaction_restores_mixed_module_modes -v) >"$out/baseline.log" 2>&1
baseline_status=$?
set -e
test "$baseline_status" -ne 0
grep -q 'FAILED (failures=1)' "$out/baseline.log"
grep -q "\['per-module modes', 'subsequent dropout forward'\]" "$out/baseline.log"

patched="$tmp/patched"
git clone --quiet --no-checkout https://github.com/mitukx/Recursive-Verification-Lag.git "$patched"
git -C "$patched" checkout --quiet --detach "$commit"
test "$(git -C "$patched" rev-parse HEAD)" = "$commit"
git -C "$patched" apply "$repo/contributions/rvl-module-mode-snapshot-v4.patch"
git -C "$patched" diff --check
(cd "$patched" && "$python" -m unittest tests.test_mini_lab_torch -v) >"$out/patched.log" 2>&1
grep -q 'Ran 13 tests' "$out/patched.log"
grep -q '^OK$' "$out/patched.log"

printf 'baseline_status=%s\npatched_status=0\nbase_commit=%s\n' "$baseline_status" "$commit" >"$out/summary.txt"
