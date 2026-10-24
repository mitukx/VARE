# Verifier-consistent asynchronous GRPO v6

v6 excludes seed 82105 after the candidate update and before signed evaluation: the negative-control evaluation trust root had an undersized run ID. The partial data remain in the v5 results directory; do not rerun the seed or use its rewards, candidate, or arm results.

Seed 82106 is the sole pilot. Run seed 82107 only if independent auditing finds at least one stale reward disagreement, VARE quarantines all eight stale members, and the fresh group is nondegenerate. The naive negative control uses pinned RVL's prompt-only reward normalization and deliberately ignores verifier-version lineage; VARE's guarded path remains unchanged.

The protocol-pinned runner and source inventory are frozen at commit
`69e9ab9b22da16413dbeababa0552e36aa06612f`. Current review fixes intentionally
change the runner/auditor hashes, so reconstruct the frozen execution worktree
for seed generation and the pilot continuation decision:

```sh
git fetch origin codex/verifier-consistent-async-grpo-v1
git worktree add --detach ../vare-v6-frozen-execution \
  69e9ab9b22da16413dbeababa0552e36aa06612f

cd ../vare-v6-frozen-execution
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82106 \
  --output results/v6-reproduction/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/v6-reproduction/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/v6-reproduction/audit-82106.json
```

Check `expand_after_pilot` before seed 82107; if true, use the same commands with seed/output `82107` and audit both eligible seeds. Run the five actual SIGKILL cutpoints only after a complete eligible pilot.

This is a synthetic verifier-drift and single-host recovery experiment, not coding capability, natural verifier-drift prevalence, or broad asynchronous throughput. Preserve all partial output directories; never overwrite one.

The completed two-seed audit and five-cutpoint process-recovery result are summarized in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md). The final independent replay is retained as `audit-with-crash-replay-order-v6.json` in the local results tree. The corrected auditor now validates the seed hash-map and crash artifact-inventory formats, requires exactly one record for each kill cutpoint, and rejects non-finite replay differences. The hash-bound source corrections are [`auditor-correction-v3.json`](auditor-correction-v3.json) and [`matched-runner-correction-v1.json`](matched-runner-correction-v1.json); the supplemental [`pilot-artifact-index-v3.json`](pilot-artifact-index-v3.json) binds the final audit and review corrections to the original raw-run inventory. The crash harness applies frozen VARE replay priority ordering and is bound by [`crash-runner-correction-v2.json`](crash-runner-correction-v2.json); [`crash-runner-correction-v3.json`](crash-runner-correction-v3.json) is test-environment portability only, while the measured run remains bound to v2. The original failed attempt remains excluded and preserved.

For a new full reproduction, use crash-harness commit `79b3d2fb324133d2a6fb738d431f97fcf7cde269` in a second worktree, produce the crash bundle from the eligible seed-82106 output using the v3 portability manifest, then run the current independent auditor with all three correction manifests. The full commands and output locations are recorded in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md). Use fresh result paths; do not overwrite retained evidence.
