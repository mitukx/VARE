# Independent seeded pass@k estimand review

Reviewed the v2 source at `3eb6de55251406ed6c8f77eeef34817f41fc5155`. All executed diagnostics used hand-made expressions (`1 + 1`, `2 + 2`) and synthetic correct/incorrect output strings. No confirmation questions or actual model output files were inspected for these diagnostics; no training or model generation was run.

## Target estimand

For each fitted run s and task t, with four responses and c correct responses, compute `1 - C(4-c,k)/C(4,k)`. Average over runs and tasks with equal weights. This represents selecting a fitted run uniformly once and attempting k responses under that fixed model. The unbiasedness interpretation assumes IID response draws conditional on that fitted model and task.

Pooling responses from five fitted runs first instead represents subsets drawn from a heterogeneous finite response pool. It is a different budget/model-selection procedure. It must not be described as the same fixed-policy pass@k. With fixed balanced sampling per run it is also not automatically an unbiased IID-mixture estimator.

## Executed synthetic counterexamples

`estimand_checks.py` and its JSON show:

- [4,0,0,0,0]: legacy pooled pass@4 = 0.6243550052, per-run mean = 0.2.
- [1,1,1,1,0]: same pooled value but per-run mean = 0.8.
- [1,1,1,0,0]: pooled = 0.5087719298, per-run mean = 0.6.

Thus the concentrated first case outranks the third under pooling, while the fixed-run target reverses the ranking.

All 3,125 possible five-run count vectors establish exact rational pass@1 equivalence under the balanced four-response grid. Float arithmetic can differ by up to 5.551115123125783e-17. Preserve the original v2 primary interval and decision unchanged, rather than assuming recomputation is byte-identical.

A second executed `_pass_at` probe gives 0.5 for one always-correct and one always-wrong hand task, but 0.6666666667 after repeating the correct task ID. New analysis should reject duplicate task IDs rather than silently reweighting them.

## Validation requirements for new analysis

- Exactly four unique sample indices per arm/checkpoint/split/task/run seed.
- Complete frozen set of run seeds; reject missing or unknown seeds.
- No duplicate progress row identity.
- Nonempty task sets with unique IDs within each split.
- Prefer globally disjoint confirm/pack IDs because the runner task lookup is keyed only by ID.
- Missing observations must fail, not shrink denominators or select complete-case subsets.
- Independently regrade raw output text rather than trust stored correctness flags.
- Shuffle/order changes must not alter weighting; task order is relevant only to deterministic bootstrap draws and frozen sampling-seed identity.

## Uncertainty and labeling

The task bootstrap of seed-averaged per-task contrasts is an analysis conditional on the fixed fitted runs. It does not establish uncertainty over retraining with new run seeds. Decoding noise is embedded in observed task summaries and is not separately identified by this bootstrap.

Across-run contrast SD includes finite decoding noise. It is not pure training randomness or automatically the standard error of the primary contrast. The base arm uses one unchanged model, so variation across its run-seed blocks is decoding variation rather than variation between trained models.

Keep legacy McNemar explicitly tied to task-level any-of-20 success over the model pool. It is distinct from the primary pass@1 contrast. Do not stack all run/task pairs and treat them as independent McNemar units; tasks are shared across runs.

Freeze the new analysis definition before any new output is observed. Old nonempty progress must not be retroactively presented as having a pre-output analysis pin. Retrospective summaries can be supported with explicit retrospective provenance while preserving original v2 summary/decision bytes.
