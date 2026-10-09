# BoolQ post-training development v17 report

## Question

The v16 study showed that a higher learning rate could lower verifier-preference loss while reducing task performance or breaching the KL cap. v17 tested whether a tenfold lower rate would retain preference fitting without degrading passage-grounded Yes/No accuracy. The protocol was frozen before these new rows were opened.

## Frozen setup

Three matched arms used the same cached Qwen2.5-0.5B-Instruct model, rank-16 output-head residual adapter, 128 training rows, 256 validation rows, and seeds 6301, 6311, and 6317. The arms were sequence DPO with base-rollout rejects, answer-only SFT, and DPO plus a 0.05 answer-likelihood anchor. The shared AdamW learning rate was 1e-5; checkpoints were fixed at epochs 0, 1, 2, 4, and 8. Runs used CPU only, offline cached assets, and no paid compute.

Training hash ranks were 152–279; validation ranks were 768–1023. The 24-row format pilot and v16 training rows were excluded. Confirmation ranks 1024–1535 were not opened. The advancement rule required base exact match >=128/256, base balanced accuracy >=0.50, mean balanced-accuracy gain >=0.05, and at least two of three seeds no worse. Selection used this development validation set, so it is not independent confirmation.

Frozen protocols: [DPO](../protocols/cpu_lm_boolq_posttraining_development_v17_dpo.lock.json), [SFT](../protocols/cpu_lm_boolq_posttraining_development_v17_sft.lock.json), [anchored DPO](../protocols/cpu_lm_boolq_posttraining_development_v17_dpo_sft_anchor.lock.json). The paired result is in the [comparison JSON](../results/cpu-lm-boolq-posttraining-development-v17-comparison.json); complete run bundles are retained [here](../results/cpu-lm-boolq-posttraining-development-v17-dpo/run-1/), [here](../results/cpu-lm-boolq-posttraining-development-v17-sft/run-1/), and [here](../results/cpu-lm-boolq-posttraining-development-v17-dpo-sft-anchor/run-1/).

## Results

The base model scored 163/256 exact matches (63.67%) and 66.84% balanced accuracy. It parsed every output as Yes/No. The validation contained 157 Yes and 99 No examples: Yes accuracy was 83/157 (52.87%) and No accuracy was 80/99 (80.81%).

| Arm | Selected epoch | Mean Exact Match | Mean balanced accuracy | Preference NLL | Preference accuracy | Mean token KL | Runtime | Peak RSS | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DPO | 2 | 63.67% | 66.84% | 0.68502 | 68.75% | 0.000113 | 1,345.8 s | 2.81 GiB | non-pass |
| Answer SFT | 2 | 63.80% | 66.94% | 0.67257 | 68.75% | 0.000400 | 1,004.2 s | 3.42 GiB | non-pass |
| Anchored DPO | 2 | 63.67% | 66.84% | 0.68081 | 68.75% | 0.000258 | 1,292.5 s | 3.42 GiB | non-pass |

DPO and anchored DPO reproduced the base predictions in all three seeds. SFT gained one correct answer in one seed (164/256) and matched base in the other two. Its mean balanced-accuracy gain was 0.00106 (0.11 percentage points), far below the frozen +0.05 requirement. All three methods therefore failed the advancement gate despite improved preference NLL. At epoch 8, SFT's KL was 1.598 and the checkpoint was ineligible; the selected epoch-2 checkpoints were within the 0.5 KL cap.

The paired question comparison found that, for one question, SFT changed one seed from incorrect to correct while DPO and anchor matched base in all seeds; the other 255 questions were tied across arms. The descriptive question-bootstrap 95% interval for DPO minus SFT and anchor minus SFT was [−0.00391, 0.00000]. Anchor and DPO tied on all 256. Because checkpoints were selected on these same validation questions, these intervals are descriptive and do not establish an independently tested difference.

## Audit and compute

All three independent run audits passed. Each reconstructed the locked 128/256 sample, verified all 15 adapter checkpoints, recomputed the selected epoch and non-pass decision, and matched three retained greedy examples to Hugging Face generation. The paired comparator verified common train and validation rows and identical base generations before comparing arms. Runs were offline and CPU-only; peak RSS stayed below 3.5 GiB.

## Interpretation and limits

The lower rate controlled policy drift, but did not produce a material task gain. DPO and anchored DPO improved verifier-preference metrics without changing validation task predictions; SFT changed one answer in one seed. This is a small, development-only result on a public benchmark and a cached 0.5B model. BoolQ may have appeared in pretraining, and the constructed preference pairs are answer-key/verifier labels rather than human preferences. The study does not establish broad reasoning, truthfulness, alignment, general post-training efficacy, downstream capability, human-preference quality, or performance at larger scale. No confirmation data was opened.

The current evidence still does not support a stable held-out model task improvement. The next useful step should target the signal or evaluation mechanism on fresh rows rather than treating this result as a learning-rate win. Any future protocol must remain separately frozen, and the reserved confirmation cohort stays unused unless a development candidate clears its gate.
