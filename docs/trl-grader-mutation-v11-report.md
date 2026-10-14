# TRL grader derived-alias and mutator audit (protocol v11)

Independent reviews of protocol v10 found accepted source mutations through derived tensor aliases, loop and context-manager targets, mutating function APIs, and aliases of the checked `normalizer`. A fresh review then found gaps for starred unpacking, structural pattern captures, generators, and comprehensions. V11 adds conservative local alias tracking and detects storage and in-place writes through these forms. The v8–v10 evaluator bundles and their retained results remain available as historical evidence.

Calibration used the pinned upstream TRL revisions. The pre-fix source failed with maximum absolute normalizer error `12.0`; the fixed source passed all 12 conditions across the main and experimental branches with maximum error `0.0`.

The pinned-source mutation audit compared v10 and v11 across 34 controls and mutations. V10 accepted 30 newly added mutation forms that v11 rejects:

| Candidate | v10 | v11 |
| --- | --- | --- |
| Fixed source control | pass | pass |
| Early return and zeroed policy numerator | reject | reject |
| Direct tuple alias storage write | reject | reject |
| Starred tuple alias, `.data`, `.view(...)`, `.detach()`, and dictionary-derived alias writes | false accept | reject |
| Pattern-capture aliases of `loss` and `normalizer` | false accept | reject |
| Generator, list-comprehension, and dictionary-comprehension derived aliases | false accept | reject |
| `for` target and `with ... as` alias writes | false accept | reject |
| `normalizer` alias writes in the policy branch and auxiliary-loss adjustment | false accept | reject |
| Alias writes inside the permitted entropy adjustment | false accept | reject |
| `operator.setitem`, imported `setitem`, and unbound `torch.Tensor.__setitem__` | false accept | reject |
| Module-level alias of `operator.setitem` | false accept | reject |
| Module-level mutator alias assigned inside top-level control flow | false accept | reject |
| Module-level mutator alias selected by a conditional expression | false accept | reject |
| Module-level mutator alias selected by a Boolean expression | false accept | reject |
| Module-level mutator alias resolved by `getattr` | false accept | reject |
| Module-level mutator alias resolved by `builtins.getattr` | false accept | reject |
| Module-level mutator alias selected from `operator.__dict__` | false accept | reject |
| Module-level mutator alias selected with `operator.__dict__.get(...)` | false accept | reject |
| Tuple-valued `out=` write through a loss alias | false accept | reject |
| Call-result `out=` write through a dictionary-held loss alias | false accept | reject |

The audit ran on macOS arm64 with Python 3.12.12 and Python's standard library. It downloaded no model weights and used no GPU, paid API, or external compute. Focused regressions also verify that unrelated pattern captures/comprehensions and the exact pinned auxiliary-loss adjustment remain accepted. V11 conservatively treats some derived copies, including `loss.clone()`, as aliases; this can reject a harmless mutation even when storage is independent.

Reproduce with:

```bash
python3.12 scripts/calibrate_task.py \
  --task-root benchmarks/historical/trl_grpo_accumulation_scale_v11 \
  --output /tmp/vare-trl-calibration-v11
python3.12 scripts/audit_trl_grader_mutations_v11.py \
  --output /tmp/vare-trl-grader-mutation-v11
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.12 -m pytest tests/test_graders.py -q
```

The calibration snapshot and raw grades, mutation patches, manifests, and result matrix are retained under [`results/trl-grpo-accumulation-window-normalizer-v1/`](../results/trl-grpo-accumulation-window-normalizer-v1/). The [locked task](../benchmarks/historical/trl_grpo_accumulation_scale_v11/) records the precise source-level contract and claim limits.

This is bounded source-structure evidence for one historical task, not arbitrary-code soundness, broad precision or recall, full trainer correctness, gradient correctness, or model capability.
