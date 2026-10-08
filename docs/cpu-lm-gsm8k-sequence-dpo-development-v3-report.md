# GSM8K sequence-level DPO development v3

## Decision: non-pass

The lower learning rate kept the selected checkpoint inside the frozen KL ceiling, but the promotion rule rejected it because free-form exact-match fell below baseline. At four epochs, mean verifier-labeled preference NLL changed from **0.69315** to **0.62001**, mean preference accuracy was **0.77083**, and mean token KL was **0.25175**. Greedy exact-match was **1/64** at base and **1/64, 0/64, 1/64** across the three seeds (updated mean **1.04%**, below the 1.56% base rate).

The frozen development decision is **`sequence_dpo_development_non_pass`**. No confirmation rows were opened.

## Protocol and execution

- New, disjoint GSM8K `train` hash ranks 2336–2399 for 64 updates and 2400–2463 for 64 validation questions.
- Qwen2.5-0.5B-Instruct, three seeds (1523, 1637, 1741), custom rank-4 frozen-backbone output-head adapter, two preference completions, CPU and offline.
- Learning rate 0.0002; update budget selected from 1, 2, 4 and 8 epochs under the 0.5 KL cap.
- Runtime: **236.66 seconds**; peak RSS: **3.42 GB**.
- The [independent auditor](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py) passed hash-rank reconstruction, artifact checks, selection/decision recomputation, and three-prompt Hugging Face generation parity.

See the [locked protocol](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v3.lock.json) and [audited run](../results/cpu-lm-gsm8k-sequence-dpo-development-v3/run-1/). The separate [confirmation v2](cpu-lm-gsm8k-sequence-dpo-confirmation-v2-report.md) had improved exact match on 128 held-out questions but failed its KL ceiling; neither result is pooled with this development set.

## Limits

All preferences come from GSM8K answer keys and a synthetic one-unit numeric distractor. The model and cohorts are small, exact-match is low, and the public dataset may have appeared in pretraining. Lowering the rate controlled KL but did not preserve generated-answer accuracy. No capability, human-preference or transfer claim is supported.
