# Verifier-consistent asynchronous GRPO v3

v3 supersedes v2 after seed 82102 stopped following its synchronous arm because token-count reporting indexed an `Attempt` object as a mapping. The partial bundle is retained under the v2 results directory and excluded. Do not rerun 82102. No stale-group result from that seed is used to select v3's pilot.

Seed 82103 is the sole pilot. Consume seed 82104 only when the independent auditor reports at least one stale-label disagreement, VARE quarantines all eight stale trajectories, and the fresh group is nondegenerate. Do not resample or change prompt, rubric, ordering, or budget.

Use the pinned local model and RVL checkout recorded in `protocol.json`:

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82103 \
  --output results/verifier_consistent_async_grpo_v3/seed-82103 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v3/seed-82103 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/verifier_consistent_async_grpo_v3/audit-82103.json
```

Check `expand_after_pilot` before using seed 82104. If true, use the same commands with seed/output `82104`; audit both eligible seed bundles together. Run the five process-kill recovery cases only after a completed eligible pilot:

```sh
python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v3/seed-82103 \
  --seed 82103 \
  --output results/verifier_consistent_async_grpo_v3/crash-82103 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned
```

This is a synthetic verifier-drift and single-host recovery experiment, not coding capability, natural drift prevalence, or broad asynchronous throughput. Preserve failed and partial output directories; never overwrite one.
