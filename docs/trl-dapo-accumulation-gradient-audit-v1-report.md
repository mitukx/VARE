# TRL DAPO accumulation-gradient audit v1

**Status:** completed, bounded source-derived gradient audit.  
**Protocol:** [`trl_dapo_accumulation_gradient_audit_v1.lock.json`](../protocols/trl_dapo_accumulation_gradient_audit_v1.lock.json)  
**Raw bundle:** [`run-1`](../results/trl-dapo-accumulation-gradient-audit-v1/run-1/)  
**Runner:** [`audit_trl_dapo_accumulation_gradient.py`](../scripts/audit_trl_dapo_accumulation_gradient.py)

## Question and prior work

TRL issue [#5619](https://github.com/huggingface/trl/issues/5619) reported that DAPO/CISPO/VESPO normalization did not account for the relation between `steps_per_generation` and the current gradient-accumulation window. The issue was fixed in merged [PR #6024](https://github.com/huggingface/trl/pull/6024). VARE already retained a source-structure grader for that fix. This follow-up asks what the locked normalizer change does to the gradient scale in a controlled, exact comparison; it is an independent quantitative reproduction of an existing upstream fix, not a new algorithm or defect report.

The source files are pinned to the issue's baseline revision `8697378709102608e3c9dc5bf582772e4ddee788` and the PR revision `93929baef2bc635d07460769cef3ce3329ea1c30`. Full-file SHA-256 values are in the protocol and run bundle.

## Estimand and method

At frozen parameters and `world_size = 1`, let `N` be the total number of valid completion tokens in the generation batch, `S` the number of microbatches per generation, and `A` the number of microbatches in the current accumulation window. For a window with masked-token gradient numerator `G_w`, the pinned baseline computes `G_w / N`. PR #6024 changes the train-mode denominator to `N * A / S`, giving `G_w / (N * A / S)`.

There are `S / A` complete windows. The primary metric is their arithmetic mean gradient divided by the full-generation masked-token mean gradient, `(sum_w G_w / N)`. Under this explicitly stated estimand, the baseline ratio is `A / S`, while the PR expression has ratio `1`. This checks the normalization semantics at fixed parameters. It does not model the parameter changes between optimizer steps.

The locked input contains four microbatches with unequal token counts (4, 8, 2, 6) and a fixed scalar gradient contribution for each token. CPU float64 autograd evaluates the pinned expressions for `A ∈ {1, 2, 4}` with `S = 4`. A separate pure-Python `Fraction` implementation independently recomputes the exact ratios without importing the runner.

## Results

| Accumulation `A` | Baseline / full-generation gradient | PR #6024 / full-generation gradient |
| ---: | ---: | ---: |
| 1 | 0.25 | 1.00 |
| 2 | 0.50 | 1.00 |
| 4 | 1.00 | 1.00 |

The autograd ratios matched the predeclared values within `1e-12`; the independent rational-arithmetic check returned exactly `1/4`, `1/2`, and `1` for the baseline and `1` for the corrected expression. The equal-window control was unchanged. The fixed source applies its rescaling only in train mode, so the eval normalizer remained unchanged.

## Reproduction

From the repository root, fetch the two exact source files and run the CPU audit:

```bash
curl -L --fail https://raw.githubusercontent.com/huggingface/trl/8697378709102608e3c9dc5bf582772e4ddee788/trl/trainer/grpo_trainer.py -o /tmp/trl-grpo-base.py
curl -L --fail https://raw.githubusercontent.com/huggingface/trl/93929baef2bc635d07460769cef3ce3329ea1c30/trl/trainer/grpo_trainer.py -o /tmp/trl-grpo-fix.py
python scripts/audit_trl_dapo_accumulation_gradient.py \
  --baseline-source /tmp/trl-grpo-base.py \
  --fixed-source /tmp/trl-grpo-fix.py
```

The runner rejects source files whose full-file hashes or expected normalization statements differ. The retained run used Python 3.9.6, PyTorch 2.8.0, CPU, and float64; it requires PyTorch but no model weights, network inference, GPU, or paid service.

## Claim boundary and decision

This establishes the exact scale consequence of the two pinned source expressions on fixed per-token gradients. It does **not** run `GRPOTrainer`, exercise Accelerate/DistributedDataParallel, execute an optimizer step, measure the evolving multi-step training trajectory, or evaluate reward/task success. The input gradients are controlled values, not sampled model tokens. Thus this is stronger than the prior AST-only source contract but remains a bounded loss-gradient audit rather than evidence of model improvement.

**Decision: stop this as a standalone research line.** The bug is already fixed upstream, and this controlled audit validates the correction's scale under its stated estimand. A further contribution is justified only by a new concrete gap—such as an end-to-end CPU regression that distinguishes the pinned source behavior through the trainer—or a different falsifiable post-training question. Do not extend this result into a capability claim.
