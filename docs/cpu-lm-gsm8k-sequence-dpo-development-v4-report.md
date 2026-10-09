# Sequence-level DPO development v4

## Decision

**Non-informative; do not advance to confirmation.** The runner's frozen rule labeled epoch 2 a development candidate because exact-match was tied at zero. That rule was too permissive at a zero-accuracy floor. The base and all three updated seeds scored 0/16 on free-form exact match, so no generation improvement was measured.

## Protocol and outcome

The locked CPU-only protocol used 16 fresh GSM8K train rows for updates and 16 for validation, three seeds, full answer-key rationales as chosen completions, and frozen base-model greedy outputs as rejected completions. Incorrect or unparseable base outputs formed 32/32 comparisons; no base output was already correct. The run used 1,008.37 seconds and peaked at 4.84 GB RSS.

At epoch 2, mean verifier-preference NLL fell from 0.69315 to 0.64854 and mean token KL was 0.000170. All three seeds had preference accuracy 1.0 on this tiny validation set. These are verifier-derived sequence-fit signals only. The free-form generation gate did not pass: base exact match was 0/16 and updated exact match was 0/16 for each seed.

Generation parity matched Hugging Face on all three audited examples. Two examples were cut off before a terminal numeric answer at the 160-token generation limit. This, together with the zero baseline, makes exact-match outcomes non-informative rather than evidence of no effect or of capability.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v4.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v4.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v4/run-2/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v4/run-2/)
- Auditor: [`scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

The audit reconstructed hash-ranked rows and rejection provenance, checked artifact hashes and adapter arrays, recomputed checkpoint selection and the frozen decision, and confirmed three generated responses against Hugging Face generation. The audit verifies implementation and records; it does not make the zero-accuracy result useful.

Run 1 is retained as a setup failure: the initial protocol was missing generation settings and stopped before model inference. It is separate from the completed run 2.

## Limits and next step

This run uses public benchmark data that may have appeared in pretraining, verifier-generated comparisons rather than human preferences, and a very small sample. It does not demonstrate free-form improvement, RLHF alignment, or generalization. The next development protocol increases the generation budget and requires a nonzero baseline plus at least one additional correct answer across the validation cohort before it can advance. It uses fresh rows and does not reopen v4 data.
