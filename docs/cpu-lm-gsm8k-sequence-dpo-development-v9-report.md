# Sequence-level DPO development v9

## Decision

**Non-pass; do not advance to confirmation.** The base scored 6/128 (4.69%). At the checkpoint selected by the frozen preference-NLL/KL rule, updated exact-match was 5/128, 4/128, and 5/128 across the three seeds (mean 4.67/128). The update lowered exact match on this cohort.

## Protocol and outcome

The run used 64 fresh hash-ranked GSM8K train questions and 128 disjoint validation questions, three seeds, and a rank-8 output adapter. Each chosen answer was the verifier's numeric answer. Rejected completions came from the frozen base model when its generated final number was wrong or unparseable; correctness-triggered one-unit counterfactuals are separately labeled. The comparisons are answer-key-derived and are not human preferences.

Epoch 2 was selected under the frozen development rule. Mean verifier-preference NLL fell from 0.69315 to 0.68039; held-out verifier-preference accuracy was 0.5833 and mean token KL was 0.20290. Free-form exact-match declined from 6/128 to a three-seed mean of 4.67/128. Epoch 4 exceeded the 0.5 KL ceiling (1.12812) and had worse mean preference NLL than the base. Thus sequence preference fit and task accuracy moved in different directions.

The independent offline audit passed. It reconstructed the rows, verifier labels, raw rollout rejections, 12 adapter artifacts and selection; all three parity generations matched Hugging Face. Runtime was 431.29 seconds and peak RSS was 3.42 GB, CPU only.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v9.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v9.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v9/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v9/run-1/)
- Auditor: [`scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

## Limits and next step

This is a small development result on a public math training split. The 128 validation questions are used to select the adapter epoch, so they cannot serve as independent confirmation. The next protocol will evaluate every eligible checkpoint on a fresh development cohort and select by verifier-checked exact match under the same KL cap, then require a new locked confirmation before any improvement claim. No broad capability result is established.
