# TRL grader tuple-alias mutation audit (protocol v9)

Protocol v8 accepted a source mutation that changed the actual policy loss after its checked division. The mutation used Python tuple unpacking to create an alias of the loss tensor and then scaled the shared tensor in place:

```python
loss_alias, = (loss,)
loss_alias.data.mul_(2)
```

The v8 static alias scan recognized assignments such as `loss_alias = loss`, but not unpacking assignments. Since the arithmetic oracle extracts and executes only the normalizer statements, the in-place loss update was outside the executed snippet and escaped the remaining checks.

Protocol v9 adds direct aliases created by tuple/list unpacking and chained assignment to the existing conservative local-name analysis. It lives in a separate task bundle so the v8 task, grader, and protocol lock remain unchanged. A non-mutating tuple alias remains accepted; an in-place update through that alias is rejected. Calibration on the pinned upstream revisions reproduced the expected result: the pre-fix revision failed with maximum normalizer error `12.0`, and the fixed revision passed all 12 arithmetic conditions with maximum error `0.0`.

The audit fetched TRL's pinned fixed revision `e1b2e21975994e676d9a676e3ab005e9bee38e40` and applied the same mutation independently to both production branches. Both graders accepted the unmodified fixed source. Both rejected the previously covered early-return and zero-numerator mutations. On the new tuple-alias mutation, v8 passed and v9 failed closed. The mutation patch SHA-256 was `041609cb082693277404b1ed656fdc14b6223992ec539ec93ec90153f84578a4`.

| Candidate source | v8 | v9 |
| --- | --- | --- |
| Fixed revision control | pass | pass |
| Early return before the checked branch | reject | reject |
| Zeroed policy-loss numerator | reject | reject |
| Tuple-unpacked alias mutates the checked loss in place | **false accept** | reject |

Reproduce with `python scripts/audit_trl_grader_mutations_v9.py --output /tmp/vare-trl-grader-mutation-v9`. The audit uses the public pinned source and Python's standard library; it downloads no model weights and uses no GPU, third-party Python package, paid API, or external compute. Focused regression tests are in [`tests/test_graders.py`](../tests/test_graders.py).

This closes one source-level false accept on one pinned task. It does not establish arbitrary-code soundness, general grader precision/recall, execution of the full loss or trainer, gradient correctness, or model capability. The v8 bundle remains available at [`benchmarks/historical/trl_grpo_accumulation_scale/`](../benchmarks/historical/trl_grpo_accumulation_scale/); v9 is at [`benchmarks/historical/trl_grpo_accumulation_scale_v9/`](../benchmarks/historical/trl_grpo_accumulation_scale_v9/).

Later independent review found a different false accept in v9: an indexed storage write through a tuple-unpacked loss alias. The v9 evaluator and original mutation result above remain unchanged; protocol v10 adds and audits that case in the [follow-up report](trl-grader-mutation-v10-report.md).

A later PR review found that v9's fixed-length alias parser also misses a starred target such as `loss_alias, *rest = (loss,)`. V9's evaluator and original mutation result remain historical; the v11 evaluator adds explicit starred-target handling and includes the pattern in its pinned-source audit. The v9 task brief's reproduction commands now point to the v9 bundle and its refreshed calibration is retained separately.
