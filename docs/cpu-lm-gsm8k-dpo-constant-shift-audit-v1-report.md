# GSM8K DPO constant-shift mechanism audit v1

## Purpose and boundary

This read-only, post hoc analysis asks whether the archived GSM8K DPO result's held-out NLL reduction can be described by a simple A/B logit shift. It uses only the already-consumed GSM8K train/held-out margins, labels, and original result bundle. It runs no model, optimizer, or new evaluation.

For each of the three existing seeds, the constant shift `c` is the mean of `(updated A-minus-B margin - base margin)` on the original 256 training rows. It is not fitted on the 1,319 held-out outcomes. On those held-out rows, the analysis compares the base margin, the counterfactual `base + c`, the counterfactual `base + (actual shift - c)`, and the archived actual DPO margin. A symmetric two-factor Shapley attribution divides actual NLL change between the constant and residual shifts. The two contributions sum exactly to the observed change.

This cannot recover the answer-content/position separation measured with paired option swaps in the ASDiv transfer study: the original GSM8K experiment saved only one A/B ordering per problem. The residual component is therefore **not** a semantic-reasoning estimate. This diagnostic is not a new performance result or a confirmation analysis.

Protocol: [`protocols/gsm8k_dpo_constant_shift_audit_v1.lock.json`](../protocols/gsm8k_dpo_constant_shift_audit_v1.lock.json). Original source artifacts and hashes are pinned in that lock. Analysis output: [`results/gsm8k-dpo-constant-shift-audit-v1/run-1/`](../results/gsm8k-dpo-constant-shift-audit-v1/run-1/).

## Results

| Seed | Train-only shift `c` | Base held-out NLL | Constant-only NLL | Actual DPO NLL | Shapley constant component | Shapley residual component |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 401 | −0.07585 | 0.75382 | 0.74527 | 0.74542 | −0.00854 | +0.00014 |
| 503 | −0.07273 | 0.75382 | 0.74560 | 0.74609 | −0.00822 | +0.00048 |
| 607 | −0.04286 | 0.75382 | 0.74883 | 0.74905 | −0.00499 | +0.00022 |
| Mean | −0.06381 | 0.75382 | 0.74657 | 0.74685 | −0.00725 | +0.00028 |

The archived actual mean held-out NLL change remains `−0.006970` nats/question. The constant-shift component is `−0.007248` nats, about **104%** of the net reduction; the nonconstant residual contributes `+0.000278` nats, slightly worsening NLL. The constant-only counterfactual is marginally better than the actual DPO adapter for all three seeds. The held-out labels are close to balanced by option: 649 correct-A and 670 correct-B rows.

The result is consistent with the original NLL gain being explained by a broad A/B output-margin calibration shift on this fixed-choice task. Because only one option ordering exists per item, this does not prove that the update had no content-dependent effects or that the same decomposition holds on other prompt orders.

## Reproduction and verification

```bash
python scripts/audit_gsm8k_dpo_constant_shift.py \
  --output results/gsm8k-dpo-constant-shift-audit-v1/rerun-1
```

The script first verifies the original result bundle manifest and the locked input-file hashes, then independently recomputes base, counterfactual, and actual NLL from per-item margins. It also checks that each seed's Shapley components reconstruct the archived NLL change and that the base/actual metrics reproduce the old report. No dataset file is reloaded and no new model output is generated.

A second clean output-directory replay produced a byte-identical `analysis.json` (SHA-256 `4dcc0c698a98f025f554fdd3a697a90ea986b92659bcff330d5cfeff1018f6aa`). The replay is retained under `results/gsm8k-dpo-constant-shift-audit-v1/replay-1/`.

## Decision

The original run remains a small same-dataset forced-choice NLL result; the mechanism audit substantially narrows its interpretation. It is not evidence of free-form math reasoning, generalization, or model capability gain. Together with the counterbalanced ASDiv non-transfer result, this closes the archived two-choice DPO line. Do not tune or rerun these adapters, reopen either consumed cohort, or extend this audit into a new preference variant.
