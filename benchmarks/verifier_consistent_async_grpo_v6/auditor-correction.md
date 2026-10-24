# V6 auditor correction

The frozen v6 protocol and measured run are unchanged. The original auditor's first attempt failed during numerical replay because it mutated `Task.metadata`, but VARE's `Task` is frozen. It wrote no report. The correction reconstructs the task with merged metadata and requires each run to contain the exact original auditor source snapshot whose SHA-256 is in protocol v6.

The corrected auditor SHA-256 and the unchanged v6 protocol SHA-256 are bound in `auditor-correction.json`. The corrected path exempts only the auditor source entry from the original source map after verifying the frozen source snapshot; runner and all other source hashes remain mandatory. This is an audit implementation erratum, not a change to experiment protocol or acceptance criteria.

Recompute from the unchanged pilot bundle:

```sh
python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction.json \
  --output results/verifier_consistent_async_grpo_v6/audit-82106-corrected.json
```
