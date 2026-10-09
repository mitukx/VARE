# Sequence-level DPO development v10

## Decision

**Non-pass; do not advance to confirmation.** Base exact match was 0/64, below the frozen minimum of 2/64. The checkpoint selected by verifier-checked exact match scored 2/64, 0/64, and 2/64 across the three seeds (mean 1.33/64), below the required mean gain of 1/32.

## Protocol and outcome

The run used fresh GSM8K train ranks 3040–3103 and validation ranks 3104–3167. It trained a rank-8 output-head adapter from numeric verifier completions versus actual incorrect base-model rollouts. Validation generations were retained for each checkpoint that improved preference NLL without exceeding the frozen 0.5 KL ceiling. The selected epoch maximized mean exact match, with NLL and then fewer epochs as tie-breakers.

Epoch 1 scored 1/64 for each seed. Epoch 2 scored 2/64, 0/64, and 2/64 and was selected. Mean verifier-preference NLL at epoch 2 was 0.66041 versus 0.69315 at base, with mean token KL 0.42954. Epoch 4 had mean NLL 0.92230 and mean KL 1.20712, so it was ineligible. Preference fit improved while task success remained near zero and variable across seeds.

The offline audit passed and independently reconstructed checkpoint eligibility, exact-match selection, the decision, sample and rejection provenance, artifacts and three Hugging Face generation parity examples. Runtime was 561.69 seconds; peak RSS was 3.42 GB; compute was CPU only.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v10.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v10.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v10/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v10/run-1/)
- Runner: [`scripts/run_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/run_cpu_lm_gsm8k_sequence_dpo_development.py)
- Auditor: [`scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

## Limits and next step

This development validation cohort also selected the checkpoint and cannot serve as independent confirmation. Results remain verifier-derived on a public math training split. A fresh larger validation cohort and a lower-rate, higher-rank development update are next; an exact-match gain on development alone would still require a new locked confirmation. No general capability result is established.
