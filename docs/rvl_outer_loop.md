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

For offline trajectory inspection, `ready_experiences()` preserves the complete nested `Generation` payload, the original replay group ID and size, RVL's stored behavior hash and its validation status, and the singleton verifier-configuration hash when present. It reproduces the two known pinned RVL hash encodings: the normal pending-to-ready path commits generation behavior fields (excluding latency), while the compatibility direct-`put` path commits the verified record (excluding timing fields). The bridge requires one encoding to match, rechecks each generation after reading, cross-checks duplicate group IDs, and refuses groups whose verifier version is stale relative to the singleton configuration hash.

This is a narrow integrity check, not an authenticity guarantee. The SQLite hash is not keyed, pending-to-ready hashes do not commit later reward/verifier output, and the singleton verifier configuration has no version-history mapping. RVL records also do not capture an immutable effective-generation-config digest; the behavior-distribution label alone cannot rule out every sampler mismatch. The bridge is therefore an offline E0 inspection adapter, not authorization to train on a trajectory. It remains single-turn, does not reconstruct a multi-event agent trace, and makes no loss/gradient-parity or model-improvement claim.

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
