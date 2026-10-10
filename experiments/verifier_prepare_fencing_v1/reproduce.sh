#!/usr/bin/env bash
set -euo pipefail
packet_dir="$(cd "$(dirname "$0")" && pwd)"
task_root="${1:?Usage: PYTHON=python3.12 bash reproduce.sh /new/scratch/directory}"
task_python="${PYTHON:-python3.12}"
if [[ -e "$task_root" ]]; then
  echo "Choose a new scratch directory; refusing to overwrite: $task_root" >&2
  exit 2
fi
mkdir -p "$task_root"
task_root="$(cd "$task_root" && pwd)"
git init -q "$task_root/base"
git -C "$task_root/base" remote add origin https://github.com/mitukx/Recursive-Verification-Lag.git
git -C "$task_root/base" fetch --depth 1 origin 27ebf7fe239d97504eeb960bf1407de220584dac
git -C "$task_root/base" checkout --detach FETCH_HEAD
git clone --shared "$task_root/base" "$task_root/patched"
git -C "$task_root/patched" apply "$packet_dir/rvl-prepare-fencing.patch"
"$task_python" "$packet_dir/reproduce.py" --repo "$task_root/base" --out "$task_root/baseline.json"
"$task_python" "$packet_dir/reproduce.py" --repo "$task_root/patched" --out "$task_root/patched.json"
"$task_python" "$packet_dir/trainer_connection.py" --repo "$task_root/base" --out "$task_root/trainer-baseline.json"
"$task_python" "$packet_dir/trainer_connection.py" --repo "$task_root/patched" --out "$task_root/trainer-patched.json"
(cd "$task_root/patched" && "$task_python" -m unittest tests.test_verifier_prepare_fencing tests.test_verifier_deployment tests.test_verifier_rpc tests.test_verifier_routing -v) > "$task_root/tests.log" 2>&1
"$task_python" - "$task_root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
b=json.loads((root/'baseline.json').read_text());p=json.loads((root/'patched.json').read_text())
tb=json.loads((root/'trainer-baseline.json').read_text());tp=json.loads((root/'trainer-patched.json').read_text())
assert b['failed_cases']==4 and b['violations']==21
assert p['violations']==0
assert tb['loss_gradient']>0 and tp['loss_gradient']<0
assert tp['gradient_error_to_intended']<=1e-6
assert tb['agrees_with_actual_rewards'] and tp['agrees_with_actual_rewards']
assert 'Ran 22 tests' in (root/'tests.log').read_text()
print('Reproduced: baseline fails four primary cases; patch passes all primary cases, trainer check, and 22 tests.')
PY
