#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-python3}
commit=c7e646b043cb56e5ea3c2623bb8a61e065451f72
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
clone="$tmp/rvl"

"$python" -c 'import sys, torch, transformers; assert sys.version_info >= (3, 11), "Python >= 3.11 required"; print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, Transformers {transformers.__version__}")'
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

git clone --quiet --no-checkout https://github.com/mitukx/Recursive-Verification-Lag.git "$clone"
git -C "$clone" checkout --quiet --detach "$commit"
test "$(git -C "$clone" rev-parse HEAD)" = "$commit"

git -C "$clone" apply "$repo/contributions/rvl-module-mode-test-only-v2.patch"
set +e
(cd "$clone" && "$python" -m unittest \
  tests.test_mini_lab_torch.TorchAcceptanceTests.test_hf_trainer_transaction_restores_mixed_module_modes \
  tests.test_mini_lab_torch.TorchAcceptanceTests.test_hf_trainer_restore_accepts_legacy_snapshot -v) >"$tmp/baseline.log" 2>&1
baseline_status=$?
set -e
cat "$tmp/baseline.log"
test "$baseline_status" -ne 0
grep -q 'test_hf_trainer_transaction_restores_mixed_module_modes.*FAIL' "$tmp/baseline.log"
grep -q 'test_hf_trainer_restore_accepts_legacy_snapshot.*ok' "$tmp/baseline.log"
grep -q 'FAILED (failures=1)' "$tmp/baseline.log"

git -C "$clone" reset --quiet --hard "$commit"
git -C "$clone" apply "$repo/contributions/rvl-module-mode-snapshot-v2.patch"
git -C "$clone" diff --check
(cd "$clone" && "$python" -m unittest tests.test_mini_lab_torch -v)
