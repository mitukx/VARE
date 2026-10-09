# BoolQ post-training study plan

## Why change tasks

Repeated GSM8K numeric-answer updates reduced verifier-preference NLL but did not produce a stable Exact Match gain over base. v15 also failed its minimum-baseline rule. The next study changes the evaluation task rather than retuning the same completion format.

BoolQ pairs naturally occurring yes/no questions with a passage and asks the model to answer from that passage. The original paper describes it as a reading-comprehension task that often requires inference, not just surface matching. The Hugging Face dataset card lists 9,427 train and 3,270 validation examples. [Original paper](https://arxiv.org/abs/1905.10044), [dataset card](https://huggingface.co/datasets/google/boolq).

A private format feasibility probe used 24 deterministically sampled **train** rows only. The cached 0.5B model returned a parseable Yes/No answer on all 24 and matched 16 labels. This tiny probe is not evidence and its rows are excluded from the formal training range. The official validation split was not used by the probe.

## Frozen design

v16 compares three matched methods: sequence-level DPO using verifier answers and base-rollout rejects, answer-only SFT, and DPO plus a fixed 0.05 chosen-answer likelihood anchor. All methods use the same 128 train questions, 256 validation questions, seeds, adapter, optimizer and update budget. The train sample excludes the 24 pilot rows.

Checkpoint selection uses class-balanced accuracy over strict one-word Yes/No exact match, after filtering by preference NLL and KL. The predeclared development gate requires at least 128/256 base exact matches, base balanced accuracy >=0.50, mean updated balanced-accuracy gain >=0.05, and at least two of three seeds no worse. Per-class support, per-class accuracy, parse rate, and ordinary Exact Match are also retained. Because validation selects checkpoints, any pass remains a candidate for a separately locked confirmation only.

The exact protocols were frozen before formal runs: [DPO](../protocols/cpu_lm_boolq_posttraining_development_v16_dpo.lock.json), [SFT](../protocols/cpu_lm_boolq_posttraining_development_v16_sft.lock.json), and [anchored DPO](../protocols/cpu_lm_boolq_posttraining_development_v16_dpo_sft_anchor.lock.json). Runs are CPU-only, offline, and use an already cached model. Confirmation rows remain reserved.

## Limits

BoolQ is a public dataset and may have appeared in pretraining. Its answer key is a verifier label, not a human preference. A successful result would be narrow evidence on one small-model reading task, not broad reasoning, truthfulness, alignment, or real-world capability. Dataset text and derived text artifacts carry the license/attribution in [the data notice](../data/BOOLQ-DATA-NOTICE.md).


## v17 lower-rate follow-up

v16 found a concrete mismatch: DPO lowered verifier-preference NLL but reduced held-out task accuracy, while SFT and anchored DPO exceeded the KL cap. v17 tests whether the shared learning rate was too aggressive. It keeps the same three matched objectives, seeds, rank-16 adapter, optimizer, balanced-accuracy selection, KL ceiling, and advancement gate, while lowering the learning rate tenfold from 1e-4 to 1e-5 and extending the frozen checkpoint schedule to epochs 1, 2, 4, and 8. This is a falsifiable development hypothesis; no outcome is assumed.

The v17 protocols were frozen before opening their new rows: [DPO](../protocols/cpu_lm_boolq_posttraining_development_v17_dpo.lock.json), [SFT](../protocols/cpu_lm_boolq_posttraining_development_v17_sft.lock.json), and [anchored DPO](../protocols/cpu_lm_boolq_posttraining_development_v17_dpo_sft_anchor.lock.json). Training uses hash ranks 152–279 (128 rows); validation uses ranks 768–1023 (256 rows). Prior pilot and v16 training rows are excluded. Validation ranks 256–767 remain reserved, and ranks 1024–1535 are reserved for a separately locked confirmation only if a development arm passes. The frozen gate still requires a baseline of at least 128/256 exact matches, baseline balanced accuracy >=0.50, mean balanced-accuracy gain >=0.05, and at least two of three seeds no worse. Checkpoint choice uses this development validation set, so a pass remains provisional.

Execution is local CPU-only and offline using the cached model, with a 6-GiB RSS and one-hour wall-clock cap per arm. All raw rows, rollouts, checkpoints, metrics, and independent audits are retained. These verifier-labeled pairs are not human preferences; BoolQ is public and may have appeared in pretraining.


## v17 outcome

v17 completed as a non-pass. Base scored 63.67% Exact Match and 66.84% balanced accuracy. DPO and anchored DPO retained exactly the base task predictions in all seeds. SFT averaged 63.80% Exact Match and 66.94% balanced accuracy, with one extra correct answer in one seed and no change in the other two; its +0.11 percentage-point balanced-accuracy gain missed the frozen +5-point gate. Preference NLL improved in all three selected arms, while all selected epoch-2 checkpoints remained far under the KL cap. All three run audits and the matched-row/base-generation comparator passed. See the [full report](boolq-posttraining-development-v17-report.md) and [comparison JSON](../results/cpu-lm-boolq-posttraining-development-v17-comparison.json). This lower-rate result does not establish a task gain; confirmation rows 1024–1535 remain unopened.
