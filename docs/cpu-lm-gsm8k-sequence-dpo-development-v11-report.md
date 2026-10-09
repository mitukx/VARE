# Sequence-level DPO development v11

## Decision

**Non-pass; do not advance to confirmation.** The base scored 9/128 exact-match (7.03%). Exact-match selection chose epoch 1, which scored 9/128, 8/128, and 9/128 across seeds (mean 8.67/128, 6.77%). This is a small decline, not a task-success gain.

## Protocol and outcome

The CPU-only run used 64 fresh GSM8K train rows (hash ranks 3168–3231) and 128 validation rows (3232–3359), three seeds, a rank-16 output adapter, and learning rate 0.0001. Each nonzero checkpoint that improved verifier-preference NLL and remained under the 0.5 token-KL limit was greedily generated on every validation question; epoch selection maximized exact match, then used NLL and fewer epochs as tie-breakers.

Epoch 1 had mean verifier-preference NLL 0.69004 and KL 0.000947. Epoch 2 lowered NLL to 0.68389 and had KL 0.008279, but all seeds scored 7/128. Epoch 4 lowered NLL further to 0.65814 with KL 0.13714, but exact-match fell to 5/128, 5/128, and 4/128. The measured preference fit therefore improved while free-form numeric task success did not.

The audit passed: it reconstructed hash-ranked rows, rollout-derived reject provenance, every eligible checkpoint's exact-match metrics, the selected epoch and non-pass decision, and three Hugging Face generation parity samples. Runtime was 1,163.68 seconds and peak RSS was 3.55 GB on CPU.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v11.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v11.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v11/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v11/run-1/)
- Runner and auditor: [`run_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/run_cpu_lm_gsm8k_sequence_dpo_development.py), [`audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

## Limits and next step

This is a development cohort on a public training benchmark. Checkpoint selection uses the same cohort and is not confirmation; verifier labels are not human preferences. The next study compares DPO against a matched supervised fine-tuning (SFT) baseline on the same fresh rows, seeds, rank, learning rate and update budget. This will test whether the pairwise reject signal adds value over simply imitating the verified answer. No broad capability claim is established.
