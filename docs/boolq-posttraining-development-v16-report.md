# BoolQ matched post-training development v16

## Decision

**No arm passes; do not advance to confirmation.** The base model scored 64.84% Exact Match and 67.26% class-balanced accuracy. DPO lowered preference NLL but reduced Exact Match to 61.72% and balanced accuracy to 63.50%. SFT and anchored DPO had no nonzero checkpoint under the frozen KL ceiling, so both selected the base checkpoint.

## Frozen study

The three locks were committed before formal runs. All arms use 128 train questions and 256 official validation questions from fresh hash-ranked rows, three seeds, rank-16 adapters, the same optimizer and update budget. The 24 rows from the private train-only format probe were excluded. The advancement rule requires at least 128/256 base Exact Matches, base balanced accuracy >=0.50, mean balanced-accuracy gain >=0.05, and at least two of three seeds no worse. Confirmation ranks 256–767 remain unopened.

The validation split contains 158 Yes and 98 No questions. Base accuracy was 90/158 (56.96%) on Yes and 76/98 (77.55%) on No, for 67.26% balanced accuracy and a 100% strict answer parse rate.

| Arm | Selected epoch | Exact Match | Balanced accuracy | Preference NLL | Pair accuracy | Mean token KL | Runtime / peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | — | 64.84% | 67.26% | 0.69315 | 0.5000 | 0 | — |
| DPO | 1 | 61.72% | 63.50% | 0.61137 | 0.7031 | 0.48728 | 554.47s / 3.32 GB |
| SFT | 0 (base) | 64.84% | 67.26% | 0.69315 | 0.5000 | 0 | 503.37s / 3.42 GB |
| Anchored DPO | 0 (base) | 64.84% | 67.26% | 0.69315 | 0.5000 | 0 | 506.60s / 3.43 GB |

DPO epoch 1 improved verifier-preference NLL (0.69315 → 0.61137) and pair accuracy (0.5000 → 0.7031), while its task metrics fell. All three DPO seeds had lower balanced accuracy than base: 66.94%, 57.76%, and 65.80%. DPO epoch 2 had lower NLL (0.53604) but exceeded the KL cap at 3.24773.

SFT epoch 1 had NLL 0.50021 but KL 7.10320; epoch 2 had NLL 0.47593 and KL 7.72261. Anchored-DPO epoch 1 had NLL 0.57453 and KL 1.29173; epoch 2 had NLL 0.52082 and KL 8.42400. These checkpoints were ineligible under the frozen 0.5 KL ceiling.

The paired DPO-minus-SFT Exact Match difference was −3.125 percentage points, with question-bootstrap 95% interval [−4.56, −1.82] points (2 better, 230 tied, 24 worse). This is descriptive because validation selected the DPO checkpoint. The independent three-arm comparator verified identical train/validation rows and identical base training/validation generations.

All three offline audits passed, including protocol and manifest hashes, rank reconstruction, selection/decision recomputation, class-stratified metrics, and three Hugging Face generation-parity examples per arm. The runs used cached data/model, CPU only, disabled network, no paid compute, and stayed below the 6-GiB RSS / 3600-second limits.

## Reproduction

- [DPO bundle](../results/cpu-lm-boolq-posttraining-development-v16-dpo/run-1/)
- [SFT bundle](../results/cpu-lm-boolq-posttraining-development-v16-sft/run-1/)
- [Anchored-DPO bundle](../results/cpu-lm-boolq-posttraining-development-v16-dpo_sft_anchor/run-1/)
- [Paired comparison](../results/cpu-lm-boolq-posttraining-development-v16-comparison.json)
- Frozen locks: [DPO](../protocols/cpu_lm_boolq_posttraining_development_v16_dpo.lock.json), [SFT](../protocols/cpu_lm_boolq_posttraining_development_v16_sft.lock.json), [anchored DPO](../protocols/cpu_lm_boolq_posttraining_development_v16_dpo_sft_anchor.lock.json)
- [BoolQ attribution and license notice](../data/BOOLQ-DATA-NOTICE.md)

## Limits and next step

BoolQ is public and may have appeared in pretraining. Its answer key is not human preference. This is evidence about one small model, one task and one constrained adapter—not general reasoning, truthfulness, alignment or real-world capability. The result shows the current learning rate is too aggressive for the SFT and anchored-DPO objectives under the KL cap, while DPO near the cap still regresses the task metric. The next study tests a lower learning rate on new rows, with the same matched controls and gates.
