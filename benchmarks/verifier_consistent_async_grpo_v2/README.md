# Verifier-consistent asynchronous GRPO v2

This protocol supersedes v1 only for fresh seeds 82102 and 82103. Seed 82101 is retained as an `ABORTED_SETUP` artifact and excluded because its run ID was invalid and the host's editable-install finder imported VARE from another checkout. Its generated reward labels and trajectories are not evidence.

The pilot is seed 82102. Seed 82103 is run only if the independent auditor finds at least one stale-label disagreement, VARE quarantines all eight stale trajectories, and the fresh group is nondegenerate. Do not resample or alter prompts, verifier rules, task ordering, or budgets.

Use the already pinned local model and RVL checkout recorded in `protocol.json`:

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82102 \
  --output results/verifier_consistent_async_grpo_v2/seed-82102 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v2/seed-82102 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/verifier_consistent_async_grpo_v2/audit-82102.json
```

Inspect the audit's `expand_after_pilot` field before consuming seed 82103. If true, run the same commands with seed/output `82103` (omit `--allow-partial` only when auditing the complete planned seed set). The five SIGKILL cases may use only an eligible completed pilot bundle:

```sh
python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v2/seed-82102 \
  --seed 82102 \
  --output results/verifier_consistent_async_grpo_v2/crash-82102 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned
```

The pilot and crash bundle are systems-mechanism evidence on a synthetic verifier transition, not coding capability, natural drift prevalence, or broad asynchronous throughput. Preserve every output directory, including failed or aborted runs; never overwrite one.
