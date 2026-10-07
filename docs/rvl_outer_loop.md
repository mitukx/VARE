# RVL outer-loop integration

VARE now has two integration paths for `mitukx/Recursive-Verification-Lag`.

## 1. Read-only replay control

`RVLTokenReplayReader` opens RVL's `TokenReplay` SQLite database in `mode=ro`. It reads the existing group `policy_version`, `verifier_version`, status and immutable generation payload without changing leases, rewards or consumption state.

```bash
vare rvl-inspect \
  --replay-sqlite /path/to/replay.sqlite \
  --current-policy-version 12 \
  --current-verifier-version 7 \
  --include-ready

vare rvl-plan \
  --replay-sqlite /path/to/replay.sqlite \
  --current-policy-version 12 \
  --current-verifier-version 7 \
  --max-policy-lag 2 \
  --max-verifier-lag 1 \
  --output artifacts/outer-plan.json
```

The planner separates systems debt from capability curriculum. Data outside the freshness bounds can trigger freeze/throttle/refresh controls but **cannot increase a task-family curriculum weight**.

## 2. Transactional GRPO hooks

`RVLGRPOHooks` wraps RVL's `HFLocalBackend` and `HFCausalLMGRPOTrainer`. Candidate training uses RVL token-exact behavior metadata, snapshots the incumbent before mutation, snapshots the candidate after GRPO, then restores the incumbent. Candidate evaluation temporarily installs the requested snapshot under a model lock and always restores the active champion afterward.

The VARE engine now allocates unique rollout sequence IDs before the first async yield and preserves complete rollout groups through replay sampling. This matters for GRPO: a comparison group must not be silently truncated by prioritized replay.

Run the real-model reference path from an environment with RVL dependencies installed:

```bash
PYTHONPATH=/path/to/Recursive-Verification-Lag:/path/to/VARE/src \
python examples/run_rvl_grpo.py \
  --dataset gsm8k \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --rounds 8 \
  --prompts-per-round 8 \
  --samples-per-prompt 8 \
  --output-dir artifacts/rvl-grpo-seed17
```

This runner has not been executed on a real GPU in this repository. A successful import/smoke run is L1; a held-out multi-seed public-model result is L2. Do not claim capability improvement until the locked protocol under `protocols/` is run and raw artifacts are retained.
