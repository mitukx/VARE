# Verifier-consistent asynchronous GRPO v5

v5 supersedes v4 after seed 82104 stopped at the VARE verifier-lineage guard. That is correct behavior for VARE, but it means a provenance-unaware negative control cannot train through the same guard. v5 makes the naive arm explicitly use pinned RVL's prompt-only `compute_group_advantages`, which deliberately ignores verifier version. The sync and VARE arms retain the unmodified provenance-aware path. The v4 partial bundle remains excluded; do not rerun seed 82104.

Seed 82105 is the sole pilot. Consume seed 82106 only if the independent auditor finds a stale-label disagreement, VARE quarantines all eight stale trajectories, and the fresh group is nondegenerate. Do not resample or change prompts, rules, order, or budgets.

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82105 \
  --output results/verifier_consistent_async_grpo_v5/seed-82105 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v5/seed-82105 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --output results/verifier_consistent_async_grpo_v5/audit-82105.json
```

Check `expand_after_pilot` before seed 82106. If true, run the same commands with seed/output `82106`, then audit both eligible bundles. Run actual SIGKILL recovery only after a complete eligible pilot:

```sh
python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v5/seed-82105 \
  --seed 82105 \
  --output results/verifier_consistent_async_grpo_v5/crash-82105 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned
```

The naive arm is an explicitly provenance-unaware RVL control. The experiment measures a synthetic verifier transition and single-host transactions, not coding capability, natural verifier-drift prevalence, or broad asynchronous throughput. Preserve every failed output directory; never overwrite one.
