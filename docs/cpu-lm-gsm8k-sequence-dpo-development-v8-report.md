# Sequence-level DPO development v8

## Decision

**Non-pass; do not advance to confirmation.** The base scored 0/32 exact-match, below the frozen minimum baseline of 1/32. Updated exact-match was 1/32, 0/32, and 0/32 over three seeds. One correct answer among 96 seed-question outputs is too small to support a gain claim.

## Protocol and outcome

The CPU-only run used 32 fresh train rows (hash ranks 2784–2815) and 32 held-out validation rows (2816–2847). The chosen completion was the GSM8K answer key's final numeric answer. The rejected completion was the frozen base model's generated answer when wrong or unparseable; if already correct, its final number was changed by one deterministically. All raw base responses and rejection sources were retained. No human preferences were used.

At epoch 4, mean verifier-preference NLL improved from 0.69315 to 0.68623, preference accuracy reached 0.7292, and mean token KL was 0.01065. Exact-match did not pass the nonzero-baseline and minimum-gain advancement rule. These preference-fit metrics do not substitute for independently observed task success.

The offline audit passed: it reconstructed the hash-ranked rows, verifier labels, rollout-derived rejects, all 12 adapter checkpoints and selection; it also matched three greedy outputs against Hugging Face. The run completed in 134.55 seconds with 3.68 GB peak RSS on CPU.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v8.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v8.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v8/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v8/run-1/)
- Auditor: [`scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

## Limits and next step

This is a small development run on a public training benchmark that may have appeared in model pretraining. It does not establish human preference alignment, free-form reasoning improvement, or general capability. The next development cohort is larger and uses a higher-rank output adapter to test whether the observed preference shift survives a more informative free-form exact-match sample. Confirmation remains locked until a development gate passes.
