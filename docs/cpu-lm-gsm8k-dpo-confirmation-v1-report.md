# GSM8K verifier-labeled preference update v1

## Question

Does a bounded preference update on a cached instruction model improve its choice between a mechanically verified answer and a nearby incorrect answer on held-out word problems?

The confirmation protocol was locked and committed as `a759381` before the official test split was loaded. Its SHA-256 is `917901cd5f2c399effa62818b7f15f9791092ef35bf04a858269dd0f5ead0d65`. A separate train-only development run selected 20 updates; the confirmation used new training rows and three new initialization seeds. No test metric changed the configuration.

The development protocol used 192 training and 96 validation questions from the GSM8K training split, with three seeds and checkpoints at 0, 5, 10, 20, and 40 updates. Mean validation NLL / mean KL were 0.69911 / 0.00000, 0.69570 / 0.00040, 0.69322 / 0.00151, 0.69051 / 0.00542, and 0.69141 / 0.01776, respectively. The locked rule selected the lowest-NLL eligible checkpoint, 20 updates. The complete development output is retained separately from the confirmation bundle.

## Method

- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, snapshot `7ae557604adf67be50417f59c2c2f167def9a775`.
- Data: [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k), 256 training questions selected after excluding the 288 rows reserved by the train-only development phase, and all 1,319 questions in the public test split. The exact cached Arrow file hashes and every selected source row are retained in the result bundle.
- Preference construction: each prompt presents the GSM8K verifier answer and a deterministic distractor exactly one unit above or below it. Option order is derived from the question hash. The preferred completion is the one-token label `A` or `B` for the correct option.
- Learner: the backbone and output head are frozen. A custom rank-4 residual updates only the two verified one-token output columns. It uses a full-batch DPO logistic objective, beta 0.1, SGD learning rate 0.01, 20 updates, and seeds 401, 503, and 607. This is a narrow custom implementation, not PEFT/TRL.
- Compute: local CPU, no GPU, no network, no paid services; 6 GiB peak RSS and one-hour limits. The run completed in 128.52 seconds with 3.19 GiB peak RSS.

## Result

The primary metric is the mean paired change in binary preference NLL (updated minus frozen base); negative is better.

| Seed | Base NLL | Updated NLL | Change | Base accuracy | Updated accuracy | Mean KL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 401 | 0.7538 | 0.7454 | −0.0084 | 0.4943 | 0.4951 | 0.00068 |
| 503 | 0.7538 | 0.7461 | −0.0077 | 0.4943 | 0.4966 | 0.00063 |
| 607 | 0.7538 | 0.7491 | −0.0048 | 0.4943 | 0.4951 | 0.00021 |

Across seeds, mean held-out NLL change was **−0.00697 nats per question**, with a seed-stratified paired bootstrap 95% interval of **[−0.00800, −0.00591]**. All three seeds improved and remained below the 0.5-nat KL ceiling. The frozen decision is **positive small-model preference result**. Mean preference accuracy moved only from 0.4943 to 0.4956, which is still close to chance; the result is a small probabilistic improvement, not a meaningful capability gain.

The bootstrap resamples questions within the three seed results on this fixed benchmark. It does not estimate transfer to other models, datasets, prompt styles, or preference sources.

The offline auditor verified all eight bundle files, rebuilt all preference pairs, matched the pinned train/test cache files and fingerprints, recomputed the per-seed NLL/accuracy/KL, bootstrap interval, and decision. It does not rerun model inference or the optimizer.

## Post hoc constant-shift mechanism audit (2026-10-10)

A later read-only analysis used only the archived train and held-out margins. For each seed it estimated a constant A-minus-B logit shift from the training rows, then compared that counterfactual with the actual adapter on the already-consumed held-out rows. The mean train-only shift was `−0.06381` logits. The constant-only counterfactual reached mean NLL `0.74657`, versus base `0.75382` and actual DPO `0.74685`. A symmetric Shapley decomposition assigns `−0.007248` nats to the constant shift and `+0.000278` nats to the nonconstant residual; the components sum to the archived `−0.006970` change. The constant component accounts for about 104% of the net NLL reduction, while the residual slightly worsens NLL. Each seed shows the same direction.

This is a post hoc mechanism description of a consumed evaluation set, not a new performance estimate. The original protocol scored only one A/B orientation per item, so it cannot separate semantic content from position bias as directly as a paired-swap evaluation. The residual is not a semantic-reasoning estimate. See [the read-only audit report](cpu-lm-gsm8k-dpo-constant-shift-audit-v1-report.md), [frozen analysis lock](../protocols/gsm8k_dpo_constant_shift_audit_v1.lock.json), and [analysis output](../results/gsm8k-dpo-constant-shift-audit-v1/run-1/).

## Limits and next question

The benchmark and answer key are public and may have appeared in model pretraining. Preferences are derived from an answer verifier, not people. The task forces a choice between two answer options and updates only the `A`/`B` output columns. It does not measure free-form response quality, mathematical reasoning, human preference alignment, reward-model behavior, or deployment utility. The small NLL change and near-chance accuracy make those limits material.

The prior arithmetic-choice model run remains a separate preregistered non-pass; this confirmation does not overwrite it. The next useful model-level step is sequence-level preference optimization over generated answer completions, paired with an independently held-out free-form answer metric. Keep that work on CPU and create a new development protocol and confirmation lock.

## Reproduction

```bash
python3 scripts/run_cpu_lm_gsm8k_dpo_confirmation.py
python3 scripts/audit_cpu_lm_gsm8k_dpo_confirmation.py \
  results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1
```

The result bundle is [`run-1`](../results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/); the locked protocol is [`cpu_lm_gsm8k_dpo_confirmation_v1.lock.json`](../protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json). The earlier update failure caused by a missing development-only adapter field is retained separately and was fixed before development metrics were produced.
