# Two-tier integer scoring

The strict extractor is the v1 final-line rule. The lenient extractor is pre-declared and is not a gate:

1. Use the last `\\boxed{integer}` when one exists.
2. Otherwise use the last integer token in the text.

`format_compliance_rate` is the fraction of rows whose last non-empty line matches the strict rule. `conditional_accuracy` is the fraction of those rows whose integer equals the expression. A row the strict rule rejects does not enter the conditional rate.

The lenient rates are separate. A missing integer counts as wrong in `lenient_accuracy_counting_miss_as_wrong`, and is omitted from `lenient_conditional_accuracy`.

`results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336/posthoc-lenient-diagnostic.json` applies both extractors to the retained v1 samples. The label is post-hoc. The v1 decision remains `stop_zero`. On the retained 256 samples the strict rule matches 3 rows, and none of those 3 integers is correct, so conditional accuracy is 0. The lenient rule extracts an integer from 242 rows and matches the expression on 19 rows (19/256 if a miss counts as wrong). Those 19 rows do not change the gate and do not authorize an update.
