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
