# Verifier-consistent asynchronous GRPO v4

v4 supersedes v3 after seed 82103 stopped before naive-arm training: the runner passed a keyword-only verifier hash parameter positionally. That partial bundle is retained under the v3 results directory and excluded. Do not rerun 82103. Its labels and stale-group outcomes were not inspected for v4 selection.

Seed 82104 is the sole pilot. Consume seed 82105 only if the independent auditor reports at least one stale-label disagreement, VARE quarantines all eight stale trajectories, and the fresh group is nondegenerate. Do not resample or change prompt, rubric, order, or budgets.

Use the pinned local model and RVL checkout recorded in `protocol.json`:

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82104 \
  --output results/verifier_consistent_async_grpo_v4/seed-82104 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v4/seed-82104 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/verifier_consistent_async_grpo_v4/audit-82104.json
```

Check `expand_after_pilot` before using seed 82105. If true, use the same commands with seed/output `82105` and audit both eligible bundles. Run process-kill recovery only after an eligible complete pilot:

```sh
python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v4/seed-82104 \
  --seed 82104 \
  --output results/verifier_consistent_async_grpo_v4/crash-82104 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned
```

This is a synthetic verifier-drift and single-host recovery experiment, not coding capability, natural drift prevalence, or broad asynchronous throughput. Preserve all failed and partial output directories; never overwrite one.
