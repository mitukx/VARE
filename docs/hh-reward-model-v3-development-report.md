# HH-RLHF v3 development report

> Historical stage report: the frozen development gate and its audit passed. The separately locked confirmation has since also passed; see the [confirmation report](hh-reward-model-v3-confirmation-report.md) for the complete current result.

## Result

The frozen development gate passed and its independent local replay audit passed. This is an outcome-informed follow-up to an exploratory v2 diagnostic; it is not an untouched preregistered claim. The confirmation block `[2048,2354)` remains sealed and has not been read.

| Measure | Development result |
| --- | ---: |
| Eligible unique prompt contexts | 512 |
| Mean calibrated-minus-raw pairwise NLL | −0.1921 nats/pair |
| Prompt-paired bootstrap 95% interval | [−0.2365, −0.1492] |
| Frozen practical-effect cutoff | ≤ −0.10 nats/pair |
| Gate | Pass |
| Independent score replay | Pass |

Each of the three fixed heads improved on the primary NLL contrast: −0.2824, −0.1252, and −0.1685 nats/pair. Their positive scales were 0.1220, 0.4110, and 0.1499. Accuracy was unchanged, as required by positive scalar scaling: 53.71%, 57.03%, and 55.86% for both raw and scaled margins. The equally weighted mean was 55.54%, compared with a 50% length-only baseline. Accuracy was descriptive and was not part of the gate.

Mean raw and scaled NLL were 0.8718 and 0.6797; the length-only baseline NLL was 0.6931. Mean Brier score fell from 0.2913 to 0.2430. Ten-bin ECE rose from 0.4497 to 0.4819, so this result does not establish improvement under every calibration metric. ECE is particularly unstable at this sample size and the selected-response target is always the dataset's chosen side.

## Frozen design and execution

Protocol [`vare-cpu-hh-reward-model-v3-fixed-head-calibration`](../protocols/cpu_hh_reward_model_v3.lock.json) was committed and pushed before any v3 test rows were opened. It freezes 768 exact fresh train contexts, excludes all 1,158 historical train/evaluation contexts and manual pilot contexts, uses 128 examples to fit each head and 128 separate examples to fit its scalar, and retains every eligible unique context in development `[1536,2048)`. The development block was required to contain at least 128; it yielded 512. The mean contrast is first averaged across the three fixed heads within each prompt; bootstrap resampling uses prompts, not head-prompt combinations.

The run used the already-cached Qwen2.5-0.5B-Instruct revision, Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.3, Datasets 4.4.2, and NumPy 1.26.4. It ran CPU-only and offline with no paid compute. Reported wall time was 464.3 seconds and peak RSS was 4,131,995,648 bytes under the frozen 3,600-second / 6-GiB ceilings.

The independent auditor reselected training and development contexts, re-extracted frozen features, refit each reward head and scalar, and reproduced per-example margins and the primary interval on the same pinned model/runtime and host. Audit status is `pass`; see [`audit.json`](../results/cpu-hh-reward-model-v3/development/run-1/audit.json) and the complete [development bundle](../results/cpu-hh-reward-model-v3/development/run-1/).

## Interpretation and limits

This development result supports only a candidate claim that a positive scalar fitted on separate train contexts can lower pairwise logistic NLL on fresh prompts from this one HH-RLHF split and feature extractor. The −0.10 threshold was informed by the already-opened v2 result. Three fixed heads do not estimate training-seed uncertainty, and the local audit is not external reproduction. The confirmation decision is not known. No assistant task success, policy update, RL gain, generalized reward-model quality, or frontier-scale behavior has been tested.

Do not parse confirmation rows unless the audited development gate passes; it did, so the frozen confirmation stage is now permitted. The confirmation gate and claim boundary are identical to development.
