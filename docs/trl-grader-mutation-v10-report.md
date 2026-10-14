# TRL grader alias-rooted store audit (protocol v10)

An independent review of protocol v9 found a concrete false accept. After
`loss_alias, = (loss,)`, the source `loss_alias[...] = 0` could change the
returned loss tensor while v9 still accepted the candidate. The v9 evaluator,
lock, and original mutation result are preserved unchanged. Protocol v10 is a
separate bundle that adds detection for attribute and subscript stores through
loss aliases, including nested assignment targets and augmented assignment to
the alias itself.

The v10 evaluator was calibrated against the same pinned TRL source revisions
as v9. The pre-fix source failed, with maximum absolute normalizer error
`12.0`; the fixed source passed all 12 arithmetic conditions, with maximum
error `0.0`. The follow-up mutation audit ran on both fixed production branches
(main DAPO/CISPO/VESPO and experimental DAPO):

| Candidate | v9 | v10 |
| --- | --- | --- |
| Fixed source control | pass | pass |
| Early return before the checked branch | reject | reject |
| Zeroed policy-loss numerator | reject | reject |
| Tuple-alias in-place method call | reject | reject |
| Tuple-alias indexed store (`loss_alias[...] = 0`) | **false accept** | reject |

Focused regression tests also reject attribute stores, nested tuple targets,
and augmented assignment while accepting harmless tuple-alias binding. The
full local suite passed 87 tests on macOS arm64 / Python 3.12.12. The audit
used the public pinned source and Python's standard library; it downloaded no
model weights and used no GPU, paid API, or external compute.

Reproduce with:

```bash
  python3.12 scripts/calibrate_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale_v10 \
  --output /tmp/vare-trl-calibration-v10
  python3.12 scripts/audit_trl_grader_mutations_v10.py \
  --output /tmp/vare-trl-grader-mutation-v10
```

Raw grades, patches, manifests, and calibration snapshots are retained under
[`results/trl-grpo-accumulation-window-normalizer-v1/`](../results/trl-grpo-accumulation-window-normalizer-v1/).

This is narrow source-level grader evidence on one historical task. It does not
establish arbitrary-code soundness, general grader precision or recall, full
trainer execution, gradient correctness, or model capability.
