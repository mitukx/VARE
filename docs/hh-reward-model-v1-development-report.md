# HH-RLHF reward-model v1 development report

## Decision

**Development non-pass. Confirmation stayed sealed.** The reward heads ranked the dataset's chosen answer correctly more often than a length-only baseline, but their probability estimates were substantially worse. The frozen gate required both accuracy and NLL improvement, so the result does not advance.

## Frozen study

The run followed [protocol v1](../protocols/cpu_hh_reward_model_v1.lock.json) on the pinned helpful-base portion of [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa). Three disjoint hash-ranked cohorts of 128 preference pairs trained linear Bradley–Terry heads over mean final-layer response-token features from a frozen Qwen2.5-0.5B-Instruct model. Each cohort also fit a response-length-only baseline. The held-out development cohort contains 256 eligible unique prompts selected from test source indices [0,512).

The test Arrow cache was materialized before the protocol freeze, but local records say no individual test rows were read or measured before the freeze. The committed runner uses the pinned raw gzip snapshot and reads only the fixed development block. Confirmation indices [512,1024) were not opened because the frozen gate failed. No GPU, paid service, or network access was used.

## Results

| Measure | Reward head, mean across 3 cohorts | Length-only baseline | Paired 95% interval for reward minus baseline |
| --- | ---: | ---: | ---: |
| Pairwise accuracy | 0.5846 | 0.5000 | [+0.0429, +0.1276] |
| Logistic NLL | 0.8566 | 0.6931 | [+0.0661, +0.2651] |
| Brier score | 0.2768 | 0.2500 | Secondary |
| 10-bin ECE | 0.4280 | 0.5000 | Secondary |

Accuracy improved by 8.46 percentage points on this cohort. NLL worsened by 0.1635, with the full paired interval above zero. All three reward heads beat chance accuracy (0.6016, 0.5664, 0.5859), while all three had worse NLL than the zero-margin length baseline. The intervals resample evaluation prompts conditional on these fixed heads; they do not include training-seed or dataset-sampling uncertainty. ECE uses the dataset's chosen response as the observed target for every pair.

The run took 259.2 seconds and peaked at 3,304,456,192 bytes RSS. The independent auditor returned status=pass: it rebuilt train and evaluation selections, exclusions, response-token counts, per-pair metrics, bootstrap intervals, and the frozen decision from the pinned sources. It did not rerun the transformer or refit the reward heads; the recorded margins remain produced by the runner.

## Interpretation and limits

This result supports a narrow statement: on one 256-pair subset of one public helpfulness-preference source, these small frozen-feature heads had higher pairwise accuracy than the response-length baseline, with worse probability quality under the frozen loss metric. It does not establish useful policy improvement, safer behavior, general preference alignment, reasoning, truthfulness, capability gain, or frontier-scale performance. The failed NLL gate is retained as evidence; no threshold or sample range was changed after seeing it.

## Artifacts

- [Frozen protocol](../protocols/cpu_hh_reward_model_v1.lock.json)
- [Audited development bundle](../results/cpu-hh-reward-model-v1/development/run-1/)
- [Study design and reproduction steps](hh-reward-model-v1.md)
