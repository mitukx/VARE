# Matched DPO, SFT and anchored DPO development v13

## Decision

**No arm passes; do not advance to confirmation.** Base exact-match was 2/64. DPO and SFT each averaged 2/64 at their selected checkpoints. DPO plus a fixed 0.05 chosen-answer likelihood anchor averaged 2.67/64, below the frozen requirement of at least +2/64 over base.

## Protocol and outcome

Three locks were committed before any v13 run. The arms share hash-ranked GSM8K train ranks 3488–3551, validation ranks 3552–3615, 64/64 questions, three seeds, rank-16 adapters, optimizer, and update counts. Their only declared differences are loss method and objective:

- **DPO:** verifier-chosen numeric answers versus actual base-model rollout rejects.
- **SFT:** token-mean cross-entropy on the chosen verifier answer only.
- **Anchored DPO:** the DPO loss plus 0.05 times chosen-answer token-mean cross-entropy.

The selected DPO checkpoint was epoch 2, with exact-match 2/64, 3/64, and 1/64 (mean 2/64), preference NLL 0.67974, preference accuracy 0.6042, and mean token KL 0.01000.

SFT selected epoch 1, with exact-match 3/64, 1/64, and 2/64 (mean 2/64), preference NLL 0.67214, preference accuracy 0.5208, and KL 0.03482. Epoch 2 lowered preference NLL to 0.63855 but breached KL at 1.35719.

Anchored DPO selected epoch 2, with exact-match 3/64, 2/64, and 3/64 (mean 2.67/64), preference NLL 0.64520, preference accuracy 0.5365, and KL 0.17945. It is directionally higher than base and the other two arms, but the gain is only 0.67/64 on a small validation cohort. The paired question-bootstrap 95% interval versus SFT was [−0.02083, 0.04167]; DPO was better/tied/worse than SFT on 1/62/1 questions, and anchored DPO on 2/61/1. These intervals are descriptive because the validation set selected each arm's checkpoint.

All three independent offline audits passed. The comparison script verified identical rows and raw base generations across arms. Runtime/RSS were: DPO 473.21s/3.44 GB, SFT 238.69s/3.44 GB, and anchored DPO 305.20s/3.45 GB; CPU only, no network or paid compute.

## Reproduction

- [DPO bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v13-dpo/run-1/)
- [SFT bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v13-sft/run-1/)
- [Anchored-DPO bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v13-dpo_sft_anchor/run-1/)
- [Paired comparison](../results/cpu-lm-gsm8k-sequence-dpo-development-v13-comparison.json)
- [Three frozen protocols](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v13_dpo.lock.json), [SFT](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v13_sft.lock.json), [anchored DPO](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v13_dpo_sft_anchor.lock.json)

## Limits and next step

The small public math cohort and verifier-derived labels do not establish human preference alignment or general capability. The anchor is promising only as a hypothesis: its point estimate was higher, but it missed the frozen advancement threshold and its paired interval includes no gain. A larger fresh comparison is needed before considering confirmation.
