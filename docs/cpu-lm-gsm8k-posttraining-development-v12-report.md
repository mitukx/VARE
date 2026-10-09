# Matched DPO versus SFT development v12

## Decision

**Neither arm passes; do not advance to confirmation.** Both arms used the same 64 train rows, 64 validation questions, three seeds, rank-16 initialization, optimizer and update counts. Base exact match was 2/64. SFT preserved 2/64 for each seed. DPO selected epoch 1 and scored 2/64, 1/64, and 1/64 (mean 1.33/64), below both baseline and SFT.

## Paired comparison

Both protocols were frozen before either run. DPO trained on verifier-chosen numeric answers against actual base-rollout rejects. SFT trained on the exact same chosen answers and rows but did not use rejected sequences in its loss. The raw base train and validation generations and all row records matched across arms.

At their independently selected checkpoints, DPO validation preference NLL was 0.68284, preference accuracy 0.724, and mean KL 0.00416. SFT had NLL 0.66593, preference accuracy 0.703, and mean KL 0.02810. DPO's pairwise accuracy was higher, but its NLL was worse and free-form exact match was lower. Across 64 paired questions, DPO was better on 0, tied on 63 and worse on 1. The question-bootstrap interval for mean exact-match difference (DPO minus SFT) was [−0.03125, 0]. This interval is descriptive: the validation cohort selected each arm's checkpoint.

Each run's offline audit passed, including row/lock reconstruction, rejection provenance, checkpoint selection, adapter hashes and three Hugging Face generation parity examples. DPO took 441.87 seconds and 3.83 GB peak RSS; SFT took 233.34 seconds and 3.75 GB. Both used local CPU only, with no network or paid compute.

## Reproduction

- [DPO lock](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v12_dpo.lock.json) and [audited bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v12-dpo/run-1/)
- [SFT lock](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v12_sft.lock.json) and [audited bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v12-sft/run-1/)
- [Paired comparison artifact](../results/cpu-lm-gsm8k-sequence-dpo-development-v12-comparison.json)
- [Comparison script](../scripts/compare_cpu_lm_gsm8k_dpo_sft.py)

## Limits and next step

This is one small development cohort on a public training split, with verifier-derived labels and validation-set checkpoint selection. It does not show a capability gain or human preference alignment. The result suggests DPO raises pairwise preference accuracy relative to SFT here but does not improve task success; a fresh study can test an SFT-anchored DPO loss as a regression-control hypothesis. Any development improvement still requires a separately locked confirmation.
