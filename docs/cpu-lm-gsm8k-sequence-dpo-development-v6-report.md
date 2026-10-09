# Sequence-level DPO development v6

## Decision

**Non-pass; do not advance to confirmation.** The locked rule required at least one baseline exact match and a mean improvement of at least one question across the 8-question validation set. The base and all three updated seeds scored 0/8.

## Protocol and outcome

This CPU-only run used fresh GSM8K train ranks 2704–2711 for updates and 2712–2719 for validation. Full answer-key rationales were chosen completions; frozen base-model rollouts were rejected completions. The chosen/rejected comparisons are verifier-constructed and are not human preferences.

At epoch 2, mean verifier-preference NLL fell from 0.69315 to 0.67510, preference accuracy was 1.0 on eight pairs, and mean token KL was 0.0000445. However, exact-match stayed at zero for the base and all seeds. Every base rollout was unparseable or incorrect. The three Hugging Face parity examples matched the custom decoder; the 192-token limit truncated all three before `####` and a final answer. This is a prompt-following/generation-budget limitation in the measured task, not evidence that the model's underlying math accuracy is zero.

Elapsed time was 609.92 seconds; peak RSS was 6,348,881,920 bytes, below the frozen 6-GiB ceiling of 6,442,450,944 bytes. The result is a verifier preference-fit signal only and does not meet the free-form improvement gate.

## Reproduction

- Locked protocol: [`protocols/cpu_lm_gsm8k_sequence_dpo_development_v6.lock.json`](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v6.lock.json)
- Audited result: [`results/cpu-lm-gsm8k-sequence-dpo-development-v6/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v6/run-1/)
- Auditor: [`scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py`](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py)

The audit reconstructed the hash-ranked rows and rejection provenance, checked all retained artifact hashes and adapters, recomputed the frozen selection and non-pass, and matched three manual generations to Hugging Face.

## Limits and next step

The small validation set is drawn from a public math training split and the model may have seen it during pretraining. Verifier rationales are not human judgments. A more suitable low-cost next probe is a numeric-only completion task, so the output format is both trainable and measurable within a short CPU generation budget. The next protocol uses entirely fresh rows and still requires a nonzero baseline plus a strict exact-match gain before confirmation.
