# GSM8K sequence-level DPO confirmation v2

## Decision: non-pass

On the fresh 128-question held-out cohort, all three seeds lowered verifier-labeled sequence-preference NLL. The mean paired change (updated minus base) was **−0.10195**, with a question-level paired bootstrap 95% interval of **[−0.13689, −0.06665]**. Greedy numeric exact-match rose from **1/128 (0.78%)** to **2/128, 4/128, and 2/128** for the three updated seeds (mean **2.08%**).

The mean full-vocabulary token KL was **0.73674**, above the frozen **0.5** ceiling. The protocol therefore returns **`sequence_dpo_confirmation_non_pass`**. The NLL and exact-match changes do not override the trust-region failure, and this run is not a capability pass.

## Protocol and execution

- Three new seeds (1301, 1409, 1511), four fixed epochs, 64 update questions and 128 held-out questions.
- Fresh GSM8K official `train` split hash ranks 2144–2207 and 2208–2335; confirmation v1's ranks remain excluded.
- Same cached Qwen2.5-0.5B-Instruct, CPU/offline execution, verifier-generated answer-vs-±1 preferences, and frozen output-head adapter as the development candidate.
- Runtime: **694.91 seconds**; peak RSS: **3.42 GB**. No GPU, paid service, or network access.
- The independent [auditor](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_confirmation.py) passed: it reproduced question-hash selection, all seed metrics, the 10,000-draw paired bootstrap, the non-pass decision, adapter hashes, and three base-generation outputs against Hugging Face `generate`.

See the [locked protocol](../protocols/cpu_lm_gsm8k_sequence_dpo_confirmation_v2.lock.json) and [raw audited bundle](../results/cpu-lm-gsm8k-sequence-dpo-confirmation-v2/run-1/). Confirmation v1 is a separate [incomplete attempt](cpu-lm-gsm8k-sequence-dpo-confirmation-v1-incomplete.md); it exceeded its two-hour limit after two of three seeds and contributes no inference.

## Limits and next step

The result measures answer-key-derived numeric preferences and exact match on a small public arithmetic benchmark that may have appeared in pretraining. Preferences are not human-labeled, exact-match remains low, and the KL ceiling failed. Do not describe this as general reasoning, human alignment, or broad model improvement. The next cycle needs fresh development rows to choose a smaller update under the same KL rule; these confirmation rows are now consumed and must not be used to tune the next run.
