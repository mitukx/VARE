# V6 crash-recovery harness correction

The first actual SIGKILL was delivered, but the restart failed before promotion because `independent_eval` recorded only the final task, yielding `n=1` against the preregistered minimum of four. The original partial attempt is retained at `results/verifier_consistent_async_grpo_v6/crash-82106` and is not counted as a recovery.

The correction moves the score assignment into the four-task loop. `crash-runner-correction.json` binds the corrected code to the frozen v6 protocol and proves the original runner hash from the v6 source commit. `auditor-correction-v2.json` binds the corrected auditor, validates the frozen auditor snapshot in both source runs, and allows the corrected crash-runner manifest. The v6 protocol and two valid model-run bundles remain unchanged.

A regression test verifies that the crash evaluator scores all four tasks. Reproduce the five cutpoints into a new directory:

```sh
python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --seed 82106 \
  --output results/verifier_consistent_async_grpo_v6/crash-82106-corrected \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction.json \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v2.json
```

Then audit the two seeds and crash bundle using the corrected auditor and both manifests:

```sh
python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --run results/verifier_consistent_async_grpo_v6/seed-82107 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --crash-run results/verifier_consistent_async_grpo_v6/crash-82106-corrected \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction.json \
  --output results/verifier_consistent_async_grpo_v6/audit-with-crash-corrected.json
```
