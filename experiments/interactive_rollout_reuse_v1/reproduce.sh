#!/usr/bin/env bash
set -euo pipefail
packet_dir=$(cd "$(dirname "$0")" && pwd)
run_root=${1:?Provide a new empty output directory}
python_bin=${PYTHON:-python3.12}
mkdir -p "$run_root"
if [ -e "$run_root/raw.json" ] || [ -e "$run_root/rvl" ]; then
  echo 'Use a new directory; evidence will not be overwritten.' >&2
  exit 1
fi
"$python_bin" - "$run_root" <<'PY'
from pathlib import Path
import urllib.request, hashlib, sys
root=Path(sys.argv[1])
url='https://raw.githubusercontent.com/ShopeeLLM/Spec-RL/97da99ed54d663b238c417c728bc824c79cfab6f/verl/utils/speculative_decoding.py'
data=urllib.request.urlopen(url).read()
assert hashlib.sha256(data).hexdigest()=='7e432a04c1de8844d4eb4ba85a59f347f344b4e5669c834d7dbe2f99f189b0c5'
(root/'speculative_decoding.py').write_bytes(data)
PY
"$python_bin" "$packet_dir/run.py" "$run_root/speculative_decoding.py" > "$run_root/raw.json"
git init -q "$run_root/rvl"
git -C "$run_root/rvl" fetch -q --depth=1 https://github.com/mitukx/Recursive-Verification-Lag.git ba94a9e513fdaf882cde091766007a88b9edaeee
git -C "$run_root/rvl" checkout -q --detach FETCH_HEAD
"$python_bin" "$packet_dir/trainer_check.py" "$run_root/rvl" > "$run_root/trainer-raw.json"
echo "Reproduction saved in $run_root"
