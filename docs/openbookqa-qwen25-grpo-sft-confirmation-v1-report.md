# OpenBookQA Qwen2.5-0.5B GRPO/SFT comparison — confirmation report

**Decision: STOP this exact model/task/training configuration.** The frozen hypothesis failed decisively on the 128-item confirmation subset. This is a negative small-model learning result, not evidence that GRPO cannot improve this model or that the model lacks the underlying science knowledge.

## Question and frozen protocol

The study asked whether one epoch of standard clipped, group-relative policy optimization (GRPO) on exact-answer rewards improves task success over both the unchanged model and matched-data supervised fine-tuning (SFT). The hypothesis was specific to `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775` and `allenai/openbookqa` main at revision `388097ea7776314e93a529163e0fea805b8a6454`.

The train, development, and confirmation cohorts each contain 128 examples selected by a frozen SHA-256 ordering from their respective official splits. The answer key supplies the independent four-choice grader. The v5 protocol fixed three seeds (`11, 23, 37`), one epoch, 16 optimizer steps per arm, full-parameter AdamW at `1e-5`, CPU float32, and 20,000 paired task-bootstrap replicates. SFT and GRPO use the same 128 training prompts; GRPO uses four base-policy completions per training prompt and the pinned RVL clipped token-level objective. GRPO's rollouts remain fixed through the one pass, with current-policy log probabilities recomputed. The predeclared success rule required at least a 3 percentage-point gain over both SFT and base, positive paired 95% bootstrap lower bounds, and a positive difference in at least two of three seeds.

The answer parser accepts a leading `A`–`D` followed by end-of-string, whitespace, period, or closing parenthesis. It records invalid-format outputs separately; it does not search free-form prose for a possibly correct option. The single parser correction was made using the disjoint development cohort before training and is recorded in the v5 lock. No parser or prompt changes followed confirmation access.

## Confirmation results

| Arm | Seed accuracies (11 / 23 / 37) | Mean exact-answer success | Parse rate |
|---|---:|---:|---:|
| Unchanged base (one checkpoint) | — | 45.31% | 100% |
| SFT | 45.31% / 27.34% / 27.34% | 33.33% | 100% |
| GRPO | 0.00% / 0.00% / 0.00% | 0.00% | 0% |

The GRPO-minus-SFT difference was **−33.33 points**, paired task-bootstrap 95% interval **[−40.89, −26.04]**. GRPO-minus-base was **−45.31 points**, interval **[−53.91, −36.72]**. All three seed-specific differences were negative against each comparator. The independent audit re-read the pinned test file, reselected the frozen cohort, recomputed every parsed label and answer-key match, and reconstructed both bootstrap intervals. It passed with zero false accepts.

All 384 GRPO confirmation responses were invalid under the frozen answer-action parser. This establishes output-format failure on this task contract; it does **not** establish that the model could not express the correct answer if prompted or parsed differently. The study reports no exact task successes because a non-label response is not a valid answer action in the preregistered interface.

Transformers emitted an incorrect-regex warning when loading the saved checkpoint tokenizers. A read-only follow-up audit checked all six checkpoint tokenizers against the base: chat-template hashes matched, and rendered prompts plus token IDs matched for all 128 training and 128 confirmation questions per checkpoint. It also checked all 1,536 stored training rollouts: prompt token IDs matched the base prompt encoding, and decoding response token IDs reproduced the stored response text in every case. The warning remains a software diagnostic; this audit rules out these specific template/prompt/trajectory alignment mismatches, not every possible tokenizer edge case. See [`audit_openbookqa_qwen_grpo_sft_tokenizer_v1.py`](../scripts/audit_openbookqa_qwen_grpo_sft_tokenizer_v1.py); its local machine-readable output is retained with the raw study artifacts.

On the separate development cohort, the mean four-answer-label KL from base was 1.591, 1.401, and 0.614 nats for GRPO seeds 11, 23, and 37; SFT values were 0.610, 0.638, and 0.541. This is KL after renormalizing over the four answer-label logits, not full-vocabulary sequence KL. In training, mean RVL behavior-KL diagnostic across GRPO updates was 9.417 and mean clipped-token fraction was 0.861. Most updates after the first had high clipping, with log-ratios reaching the trainer's cap of 20. These diagnostics are consistent with substantial policy drift during reuse of fixed base-policy rollouts. They do not isolate drift as the cause of the invalid outputs; no KL-penalty or rollout-refresh ablation was part of the frozen study.

## Integrity, recovery, and resources

All six SFT/GRPO checkpoints completed 16 optimizer steps and were saved before confirmation evaluation. The first v5 end-to-end runner then failed at its first evaluation-model construction with `NameError: HFLocalBackend is not defined`; it had opened the confirmation split only after training and produced no model outputs on it. The evaluator-only v6 recovery lock was frozen against the completed v5 progress file, all six checkpoint file hashes, candidate fingerprints, pinned sources, and confirmation-ID receipt. It evaluated the unchanged base and exact saved checkpoints without retraining. The checkpoint hashes and reloaded parameter fingerprints matched. The separate v5 audit independently reconstructed the final task metrics from raw responses.

The pre-training base gate passed: development exact answer was 55/128 (42.97%) at 100% parse rate. All 1,536 sampled training responses parsed; the three seeds had 235/257/234 positive traces and 66/64/61 mixed-reward groups, respectively. Thus the run did not begin with a missing or constant reward signal.

Measured CPU-only wall time was 244.7 s for the base feasibility/rollout gate, 1,067.8 s for training all six arms (the failed v5 stage includes post-training test-file loading), and 688.6 s for confirmation evaluation. Peak RSS was 11,924.6 MiB in the training stage and 4,032.0 MiB in evaluation, within the frozen 24 GiB cap. No GPU, MPS, paid API, or external spend was used.

The dataset host lists the content license as unknown. Raw dataset files, prompts, responses, and the 11 GiB of candidate checkpoints therefore remain in ignored local `work/` and `artifacts/` paths and are not included here. The protocol locks retain source revisions and hashes, not question text. A complete independent reproduction requires lawful access to the pinned dataset content and rerunning training; the committed v6 evaluator is a recovery evaluator bound to this host's exact saved checkpoints, not a portable fresh-run harness. Benchmark/model pretraining overlap cannot be excluded, and 128 confirmation items provide limited precision. No external human review or reproduction has occurred.

## Reproduction and retained artifacts

The v5 training protocol and source are [`openbookqa_qwen_grpo_sft_v5.lock.json`](../protocols/openbookqa_qwen_grpo_sft_v5.lock.json), [`run_openbookqa_qwen_grpo_sft_v5.py`](../scripts/run_openbookqa_qwen_grpo_sft_v5.py), and [`audit_openbookqa_qwen_grpo_sft_v5.py`](../scripts/audit_openbookqa_qwen_grpo_sft_v5.py). The recovery evaluator and its artifact-bound lock are [`evaluate_openbookqa_qwen_grpo_sft_v6.py`](../scripts/evaluate_openbookqa_qwen_grpo_sft_v6.py) and [`openbookqa_qwen_grpo_sft_v6-evaluation.lock.json`](../protocols/openbookqa_qwen_grpo_sft_v6-evaluation.lock.json).

On the original host, the recovery command was:

```sh
work/openbookqa-v1/runtime-v5/bin/python scripts/evaluate_openbookqa_qwen_grpo_sft_v6.py \
  --model-path "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" \
  --rvl-source results/openbookqa-posttraining-v1/source-checkouts/Recursive-Verification-Lag \
  --test-parquet work/openbookqa-v1/pinned/main/test-00000-of-00001.parquet \
  --dev-parquet work/openbookqa-v1/pinned/main/validation-00000-of-00001.parquet \
  --output artifacts/openbookqa_qwen_grpo_sft_v1/study-v6-evaluation

work/openbookqa-v1/runtime-v5/bin/python scripts/audit_openbookqa_qwen_grpo_sft_v5.py \
  --mode final \
  --rvl-source results/openbookqa-posttraining-v1/source-checkouts/Recursive-Verification-Lag \
  --model-path "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" \
  --test-parquet work/openbookqa-v1/pinned/main/test-00000-of-00001.parquet \
  --summary artifacts/openbookqa_qwen_grpo_sft_v1/study-v6-evaluation/summary.json
```

The local raw summary is `artifacts/openbookqa_qwen_grpo_sft_v1/study-v6-evaluation/summary.json`; the token-alignment audit is `artifacts/openbookqa_qwen_grpo_sft_v1/study-v6-evaluation/tokenizer-integrity-audit.json`. The original failed-v5 stage, including its exception record and trained checkpoints, remains at `artifacts/openbookqa_qwen_grpo_sft_v1/study-v5/`.

## Decision

**STOP** this exact pairing and protocol. Do not tune its learning rate, parser, prompt, seed, or confirmation cohort and do not present the result as a general GRPO failure. The primary hypothesis did not survive, and the result exposes a concrete risk: a small-data, full-parameter GRPO run can drift far from its fixed behavior policy while training reward remains nonzero, then lose the required answer format on every held-out item. The next credible model-learning claim requires a new, separately frozen cohort and a predeclared intervention that controls or tests policy drift; it must not reuse these confirmation rows.
