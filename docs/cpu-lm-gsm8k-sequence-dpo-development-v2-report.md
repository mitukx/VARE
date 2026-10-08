# GSM8K sequence-level DPO development v2

## Result

The frozen development rule selected four epochs. Across three seeds, mean validation sequence-preference NLL fell from **0.69315** at the base model to **0.57436**. Mean held-out preference accuracy rose from **0.5000** to **0.8177**, with mean full-vocabulary token KL **0.43843**, below the frozen 0.5 ceiling. The greedy free-form exact-match rate was **1/64** at baseline and **1/64** for each updated seed. This is a development candidate based on a small verifier-labeled validation set; it does not show a free-form accuracy gain.

## Protocol and execution

- Qwen2.5-0.5B-Instruct, pinned cached revision; GSM8K official `train` split only.
- 64 development update rows and 64 disjoint validation rows selected by the frozen question-hash ranking at ranks 1376–1439 and 1440–1503.
- Three seeds (109, 211, 307); four epochs selected from checkpoints 1, 2, 4 and 8 using validation NLL subject to KL ≤0.5.
- Sequence-summed DPO on a numeric answer and a deterministically generated answer one unit away. The preference labels come from GSM8K answer keys, not human judgments.
- Custom rank-4 residual over the frozen full-vocabulary output head; all model computation ran on CPU, offline, with cached weights. No GPU, paid service, or network access was used.
- Runtime: 201.57 seconds; peak RSS: 3.42 GB.

The result is reproducible from the [locked protocol](../protocols/cpu_lm_gsm8k_sequence_dpo_development_v2.lock.json) and [retained run bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v2/run-2/). The independent [auditor](../scripts/audit_cpu_lm_gsm8k_sequence_dpo_development.py) rechecks hashes, protocol lock, row separation, token suffixes, adapter files, per-seed summaries, selection, and promotion decision. It also matched the custom greedy decoder to Hugging Face generation on three validation prompts.

## Invalidated predecessor and limits

The first sequence-DPO development attempt (v1) is not evidence for generated-answer quality. Its custom batched decoder disagreed with Hugging Face generation; a parity check traced this to batching behavior and failure to apply Qwen's pinned repetition penalty. The v1 raw files remain on the local machine as an audit trail and are not included in the v2 result. Version 2 uses a one-prompt-at-a-time decoder with the pinned repetition penalty. The parity check covers three prompts, not all outputs.

This remains a small development result on public GSM8K data that may have appeared in pretraining. It does not establish an independent confirmation, human preference alignment, broad reasoning, transfer, or model capability improvement. The free-form exact-match result is unchanged. A separate confirmation on new training-split rows must be locked before those rows are read; this report must not be used as a capability claim.
