#!/usr/bin/env bash
set -euo pipefail
packet_dir=$(cd "$(dirname "$0")" && pwd)
run_root=${1:?Provide a new empty output directory}
python_bin=${PYTHON:-python3.12}
mkdir -p "$run_root"
run_root=$(cd "$run_root" && pwd)
if [ -e "$run_root/rvl" ]; then echo 'Use a new directory.' >&2; exit 1; fi
git init -q "$run_root/rvl"
git -C "$run_root/rvl" fetch -q --depth=1 https://github.com/mitukx/Recursive-Verification-Lag.git 436a96e86b704ad981b05237b41fdfb846f18817
git -C "$run_root/rvl" checkout -q --detach FETCH_HEAD
"$python_bin" "$packet_dir/reproduce.py" "$run_root/rvl" > "$run_root/baseline.json"
git -C "$run_root/rvl" fetch -q --depth=1 https://github.com/mitukx/Recursive-Verification-Lag.git ba94a9e513fdaf882cde091766007a88b9edaeee
git -C "$run_root/rvl" checkout -q --detach FETCH_HEAD
"$python_bin" "$packet_dir/reproduce.py" "$run_root/rvl" > "$run_root/patched.json"
(cd "$run_root/rvl" && "$python_bin" -m unittest tests.test_verifier_rpc tests.test_verifier_deployment tests.test_verifier_prepare_fencing tests.test_verifier_same_epoch tests.test_verifier_routing -v) > "$run_root/tests.log" 2>&1
"$python_bin" - "$run_root" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]);base=json.loads((p/'baseline.json').read_text());fixed=json.loads((p/'patched.json').read_text())
assert base[0]['wrong_artifact_success'] and not base[0]['fixed_contract_pass']
assert all(row['fixed_contract_pass'] for row in fixed)
assert base[1]['fixed_contract_pass']
print('Baseline mismatch, fixed rejection/recovery, and identical control reproduced.')
PY
