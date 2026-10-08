# Human preference reward-model study v1

## Question

Can a small scalar reward head trained on human helpfulness comparisons rank new preferred responses better than a response-length baseline, using frozen language-model features and CPU-only execution?

## Frozen design

The study uses only the `helpful-base` split from the pinned [HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa). The dataset card describes these human pairwise preferences as intended for preference/reward-model training, not supervised dialogue-agent training, and advises users to consider their own risk tolerance because examples may be offensive or upsetting. The dataset is MIT-licensed at this pinned revision.

Three disjoint training cohorts each contain 128 unique, hash-ranked training prompt contexts. A linear scalar reward head is trained with the pairwise Bradley–Terry logistic loss over mean final-layer Qwen response-token features. The transformer stays frozen. Each cohort also gets a length-only logistic baseline fitted on the same training pairs. Contexts from the six manually inspected training pilot rows are excluded from training and both evaluation cohorts.

The first 256 eligible unique prompts in official test indices `[0,512)` form development. Indices `[512,1024)` are reserved for confirmation and are not opened unless the audited development rule passes. Eligibility is checked before a context can be accepted: parse both transcripts, require matching contexts and nonempty final responses, tokenize each exact transcript with the pinned fast tokenizer using special tokens and no chat template, require each full sequence to fit within 1024 tokens and each response to contain feature tokens. Invalid or overlength rows do not reserve their context. Then skip pilot, train, and (for confirmation) development contexts; accept the first eligible occurrence in source order. If the fixed block has fewer than 256 eligible contexts, stop as a protocol failure without backfilling from another block. All ranges and thresholds are frozen in the [protocol](../protocols/cpu_hh_reward_model_v1.lock.json).

The test Arrow cache had already been materialized by the dataset library before the protocol freeze. The local project record says its individual records were not read or measured. The runner opens only the locked development index block and only reads the confirmation block after the development gate passes.

## Evaluation and claim limits

For each pair, the target is that the dataset's chosen answer is preferred (`y=1`); the predicted probability is `sigmoid(reward(chosen) - reward(rejected))`. Pairwise accuracy is 1 for a positive margin, 0 for a negative margin, and 0.5 for an exact tie. Logistic NLL is `mean(softplus(-margin))`, and Brier score is `mean((p - 1)^2)`. Differences are reward head minus length baseline, so a negative NLL difference favors the reward head. ECE uses ten fixed probability bins `[0,.1)`, ..., `[.9,1]`, weighted by bin count, comparing mean probability with the observed fraction of chosen labels (`y=1`).

The primary result compares paired preference accuracy against the length-only logistic baseline. The frozen gate also requires a 95% prompt-level percentile-bootstrap interval above zero for accuracy gain, at least a two-point gain, lower reward-head NLL with its paired interval entirely below zero, and above-chance accuracy in at least two of three training cohorts. Bootstrap resamples the same evaluation prompt indices for accuracy and NLL, averages paired differences across the three fitted heads per prompt, and reports 2.5th/97.5th percentiles over 10,000 resamples. These intervals are conditional on the fitted heads and selected cohort; they do not measure variation from retraining or resampling the dataset. Calibration and Brier score are secondary.

A pass would support only human preference prediction on this one dataset subset and frozen feature extractor. It would not show assistant quality, safety, user utility, policy improvement, reasoning, or capability gain. No GPU, paid service, model download, or transcript release is part of the design. Result bundles retain row indices, domain-separated prompt hashes, aggregate predictions and metrics, but never raw transcripts or hidden-state features.

## Reproduction

After the runner and auditor are committed, run the CPU-only development phase and audit it before opening confirmation rows:

```bash
python3 scripts/run_cpu_hh_reward_model.py --stage development
python3 scripts/audit_cpu_hh_reward_model.py results/cpu-hh-reward-model-v1/development/run-1
```

Run confirmation only when the frozen development gate passes and the development audit is recorded:

```bash
python3 scripts/run_cpu_hh_reward_model.py \
  --stage confirmation \
  --development-audit results/cpu-hh-reward-model-v1/development/run-1/audit.json
python3 scripts/audit_cpu_hh_reward_model.py results/cpu-hh-reward-model-v1/confirmation/run-1
```
