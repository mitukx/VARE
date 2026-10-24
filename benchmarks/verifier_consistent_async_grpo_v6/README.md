# Verifier-consistent asynchronous GRPO v6

v6 excludes seed 82105 after the candidate update and before signed evaluation: the negative-control evaluation trust root had an undersized run ID. The partial data remain in the v5 results directory; do not rerun the seed or use its rewards, candidate, or arm results.

Seed 82106 is the sole pilot. Run seed 82107 only if independent auditing finds at least one stale reward disagreement, VARE quarantines all eight stale members, and the fresh group is nondegenerate. The naive negative control uses pinned RVL's prompt-only reward normalization and deliberately ignores verifier-version lineage; VARE's guarded path remains unchanged.

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82106 \
  --output results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/verifier_consistent_async_grpo_v6/audit-82106.json
```

Check `expand_after_pilot` before seed 82107; if true, use the same commands with seed/output `82107` and audit both eligible seeds. Run the five actual SIGKILL cutpoints only after a complete eligible pilot.

This is a synthetic verifier-drift and single-host recovery experiment, not coding capability, natural verifier-drift prevalence, or broad asynchronous throughput. Preserve all partial output directories; never overwrite one.

The completed two-seed audit and five-cutpoint process-recovery result are summarized in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md). The crash harness applies frozen VARE replay priority ordering and is hash-bound by [`crash-runner-correction-v2.json`](crash-runner-correction-v2.json); the original failed attempt remains excluded and preserved.
