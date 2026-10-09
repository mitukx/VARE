# Matched SFT and anchored DPO development v15

## Decision

**Both arms are non-passes; do not advance to confirmation.** Base exact-match was 7/256, below the frozen 8/256 minimum baseline. SFT selected a mean of 6.33/256; anchored DPO selected 7/256, equal to base. Neither meets the required mean gain of 8/256.

## Protocol and outcome

The two protocol locks were committed before either run. Both arms share fresh hash-ranked GSM8K rows 3808–3871 for training and 3872–4127 for validation, 64/256 questions, seeds 5413/5501/5507, rank-16 adapters, optimizer and update counts. The only declared method differences are the SFT versus anchored-DPO objectives. Confirmation ranks beginning at 4128 were not opened.

| Arm | Selected epoch | Exact Match by seed | Mean Exact Match | Preference NLL | Pair accuracy | Mean token KL | Runtime / peak RSS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| SFT | 1 | 7, 6, 6 / 256 | 6.33/256 | 0.67647 | 0.5677 | 0.04452 | 674.69s / 3.61 GB |
| Anchored DPO | 2 | 8, 6, 7 / 256 | 7/256 | 0.65606 | 0.5951 | 0.15042 | 885.63s / 3.66 GB |

Anchored DPO minus SFT was +0.67/256 Exact Match, with paired question-bootstrap 95% interval [−0.00651, 0.01302] (3 better, 251 tied, 2 worse). These intervals are descriptive because the same validation set selected each arm's checkpoint. The comparator verified identical training/validation examples and identical raw base training/validation generations.

Both independent offline bundle audits passed, including protocol and manifest hashes, rank reconstruction, checkpoint selection, decision recomputation, and three Hugging Face generation-parity examples per arm. Runs used cached model/data, CPU only, disabled network, no paid compute, and stayed below the frozen 6-GiB RSS / 3600-second limits.

## Reproduction

- [SFT bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v15-sft/run-1/)
- [Anchored-DPO bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v15-dpo_sft_anchor/run-1/)
- [Paired comparison](../results/cpu-lm-gsm8k-sequence-dpo-development-v15-comparison.json)
- Frozen locks: [SFT](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v15_sft.lock.json), [anchored DPO](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v15_dpo_sft_anchor.lock.json)

## Limits and next step

This is a small public math-task development study with verifier-derived labels. It does not establish human preference alignment, broad reasoning, general language quality, or capability gain. The anchor did not improve exact-match over base on this new cohort, and the SFT control declined. Repeated small-cohort tuning is not justified by these results. The next study should change the evaluation task or feedback signal and test an explicit mechanism on fresh data; it must preserve a nonzero baseline, a meaningful gain threshold, matched controls, and an untouched confirmation boundary.
