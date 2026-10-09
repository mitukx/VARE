# Matched DPO, SFT and anchored DPO development v14

## Decision

**No arm passes; do not advance to confirmation.** Base exact-match was 4/128. DPO averaged 4.67/128, SFT 5.33/128, and anchored DPO 7.67/128. The frozen advancement threshold was a mean gain of at least 4/128, so the anchor's +3.67/128 point estimate narrowly fails. All three arms had at least two seeds no worse than base, but that secondary condition does not replace the gain threshold.

## Protocol and outcome

The three protocol locks were committed before runs. They share fresh hash-ranked GSM8K rows 3616–3679 for training and 3680–3807 for validation, 64/128 questions, seeds 5209/5303/5393, rank-16 adapters, optimizer, and update counts. Method and objective are the declared arm differences. The frozen gates require base >=4/128, mean Exact Match gain >=4/128, and at least two of three seeds no worse. Confirmation ranks beginning at 3808 were not opened.

| Arm | Selected epoch | Exact Match by seed | Mean Exact Match | Preference NLL | Pair accuracy | Mean token KL | Runtime / peak RSS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| DPO | 2 | 5, 4, 5 / 128 | 4.67/128 | 0.67061 | 0.6719 | 0.03952 | 894.94s / 3.75 GB |
| SFT | 1 | 6, 5, 5 / 128 | 5.33/128 | 0.66439 | 0.6432 | 0.03093 | 373.47s / 3.56 GB |
| Anchored DPO | 2 | 7, 7, 9 / 128 | 7.67/128 | 0.62950 | 0.6979 | 0.16201 | 486.52s / 3.76 GB |

The anchor has the highest point estimate, but it is 0.33/128 below the minimum gain rule. The paired question-bootstrap 95% interval versus SFT was [−0.015625, 0.054688], with anchor better/tied/worse on 8/115/5 questions. DPO versus SFT was [−0.03125, 0.018229], with DPO better/tied/worse on 3/121/4. These are descriptive intervals because each arm's checkpoint was selected on the same validation cohort. The comparison tool verified the same train/validation examples and identical raw base training/validation generations across all arms.

All three independent offline bundle audits passed, including rank reconstruction, protocol and manifest hashes, checkpoint selection, decision recomputation, and three Hugging Face generation-parity examples per arm. Runs used cached model/data, CPU only, disabled network, no paid compute, and stayed below the frozen 6-GiB RSS / 3600-second limits.

## Reproduction

- [DPO bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v14-dpo/run-1/)
- [SFT bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v14-sft/run-1/)
- [Anchored-DPO bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v14-dpo_sft_anchor/run-1/)
- [Paired comparison](../results/cpu-lm-gsm8k-sequence-dpo-development-v14-comparison.json)
- Frozen locks: [DPO](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v14_dpo.lock.json), [SFT](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v14_sft.lock.json), [anchored DPO](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v14_dpo_sft_anchor.lock.json)

## Limits and next step

This is a small public math-task development study with verifier-derived labels. The result is not human preference evidence, does not measure general language quality, and does not establish reasoning or capability gains. The anchor trend deserves a fresh, larger direct comparison with SFT, but this run is a non-pass and cannot justify confirmation. The next development probe should use an untouched larger validation cohort and keep its gain rule fixed before training.
