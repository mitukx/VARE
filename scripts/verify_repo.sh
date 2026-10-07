#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH=src python -m compileall -q src tests examples
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m vare.cli demo --rounds 8 --rollouts 512 --seed 7 --output artifacts/demo.json >/tmp/vare-demo.txt
PYTHONPATH=src python -m vare.cli lock-protocol --spec protocols/l2_rvl_qwen_v1.spec.json --output /tmp/vare-lock.json >/tmp/vare-lock.txt
cmp /tmp/vare-lock.json protocols/l2_rvl_qwen_v1.lock.json
rm -rf /tmp/vare-env-smoke /tmp/vare-env-campaign
PYTHONPATH=src python -m vare.cli env-smoke --output-dir /tmp/vare-env-smoke >/tmp/vare-env-smoke.txt
PYTHONPATH=src python -m vare.cli env-campaign --catalog-root benchmarks/smoke --agent-argv-json '["python","examples/oracle_smoke_agent.py","{workspace}"]' --repeats 1 --run-root /tmp/vare-env-campaign >/tmp/vare-env-campaign.txt
cat /tmp/vare-demo.txt
cat /tmp/vare-env-smoke.txt
cat /tmp/vare-env-campaign.txt
