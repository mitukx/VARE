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
git worktree add --detach ../vare-v6-corrected-auditor \
  73b62971b8fe4df6c8ec0a6fa80a67af3a784093

cd ../vare-v6-frozen-execution
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82106 \
  --output results/v6-reproduction/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

(cd ../vare-v6-corrected-auditor && python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run ../vare-v6-frozen-execution/results/v6-reproduction/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v4.json \
  --runner-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/matched-runner-correction-v1.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/v6-reproduction/audit-82106.json)
```

Do not use the frozen execution checkout's auditor for the pilot gate: its `Task` reconstruction has a retained `FrozenInstanceError`. Generate rollouts from the frozen runner, but audit them with the corrected auditor at commit `73b62971b8fe4df6c8ec0a6fa80a67af3a784093` and the hash-bound auditor, matched-runner, and crash-runner manifests shown above. The crash-runner manifest binds the corrected source even though the pilot audit has no crash bundle; crash recovery is audited only after the pilot continuation decision. Check `expand_after_pilot` before seed 82107; if true, generate the second seed with the frozen runner and audit both eligible seeds. Run the five actual SIGKILL cutpoints only after a complete eligible pilot.

This is a synthetic verifier-drift and single-host recovery experiment, not coding capability, natural verifier-drift prevalence, or broad asynchronous throughput. Preserve all partial output directories; never overwrite one.

The completed two-seed audit and five-cutpoint process-recovery result are summarized in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md). The original [`final-independent-audit-v6.json`](final-independent-audit-v6.json) and its outcome-equivalent v4 re-audit [`final-independent-audit-v6-correction-v4.json`](final-independent-audit-v6-correction-v4.json) are committed, as is the successful seed-only continuation-gate audit [`pilot-gate-audit-82106-correction-v4.json`](pilot-gate-audit-82106-correction-v4.json). The corrected auditor validates both manifest formats, requires exactly one record for each kill cutpoint, rejects non-finite replay differences, and can validate corrected runner sources in a seed-only pilot audit. [`auditor-correction-v4.json`](auditor-correction-v4.json), [`matched-runner-correction-v1.json`](matched-runner-correction-v1.json), and [`crash-runner-correction-v3.json`](crash-runner-correction-v3.json) bind the source corrections; [`pilot-artifact-index-v5.json`](pilot-artifact-index-v5.json) binds these audits and current documentation to the original raw-run inventory. The crash harness applies frozen VARE replay priority ordering and is bound by [`crash-runner-correction-v2.json`](crash-runner-correction-v2.json); v3 is test-environment portability only, while the measured run remains bound to v2. The original failed attempt remains excluded and preserved.

For a new full reproduction, use crash-harness commit `79b3d2fb324133d2a6fb738d431f97fcf7cde269` in a second worktree, produce the crash bundle from the eligible seed-82106 output using the v3 portability manifest, then run the corrected independent auditor at `73b62971b8fe4df6c8ec0a6fa80a67af3a784093` with all three current correction manifests. The full commands and output locations are recorded in [`crash-recovery-results-v6.md`](crash-recovery-results-v6.md). Use fresh result paths; do not overwrite retained evidence.
