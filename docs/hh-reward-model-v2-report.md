# HH-RLHF reward-model v2 report

## Decision

**Development passed; confirmation did not pass the joint gate.** The accuracy gain over the length-only baseline repeated on the fresh confirmation cohort. The paired NLL interval crossed zero, so the frozen requirement for improved probability quality was not met. Do not describe this as a confirmed calibrated reward model.

## Frozen study

The study follows [protocol v2](../protocols/cpu_hh_reward_model_v2.lock.json) on the pinned helpful-base split of [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa). Three disjoint hash-ranked cohorts of 128 preference pairs train linear Bradley–Terry heads over mean final-layer response-token features from frozen Qwen2.5-0.5B-Instruct weights. A length-only logistic scorer is the paired baseline. A positive scalar is fitted to two-fold out-of-fold training margins and applied unchanged to the final 128-pair head.

V1 development used source range [0,512) and failed its NLL gate. V2 excludes its 256 selected context hashes. V2 development selected 256 unique prompts from [512,1024), and confirmation selected 256 from [1024,1536). The selected source-index minima and maxima were 512–769 for development and 1024–1280 for confirmation; ineligible duplicate-context rows created gaps at indices 567 and 743 in development and 1166 in confirmation. Confirmation rows were opened only after the development audit passed. The full compressed test file was hashed as opaque bytes for provenance; the row readers did not return, parse or score confirmation records before the gate.

No GPU, network access or paid compute was used. The cached model and data revisions were pinned before the run.

## Results

| Measure | Development | Confirmation | Length baseline |
| --- | ---: | ---: | ---: |
| Pairwise accuracy | 0.5996 | 0.5768 | 0.5000 |
| Accuracy gain, paired prompt-bootstrap 95% interval | +0.0996 [+0.0560, +0.1419] | +0.0768 [+0.0313, +0.1224] | — |
| Logistic NLL | 0.6462 | 0.6738 | 0.6931 |
| NLL difference, reward minus baseline, paired 95% interval | −0.0470 [−0.0777, −0.0163] | −0.0194 [−0.0539, +0.0177] | — |
| Brier score | 0.2276 | 0.2391 | 0.2500 |
| Ten-bin ECE | 0.4490 | 0.4605 | 0.5000 |

All three heads had accuracy above 0.5 on both cohorts. Two of three confirmation heads had lower NLL than the baseline; the third was slightly worse. The confirmation NLL interval includes zero, so the mean reduction is inconclusive under the frozen paired criterion. The gate requires both a clear accuracy gain and NLL improvement; confirmation is therefore a non-pass despite the repeated accuracy signal.

Development used 260.7 seconds and peaked at 3,099,033,600 bytes RSS. The confirmation runner, including its fresh development audit, used 534.4 seconds and peaked at 3,565,743,872 bytes RSS. Both remained within the 3,600-second and 6-GiB limits.

## Audit and provenance

The development and confirmation bundles each have an independent audit with status `pass`. The auditor rebuilt row selection and exclusions, retokenized the pinned text, re-extracted frozen-model features, refit the heads and calibration, and reproduced per-pair margins within 2e-4. It uses the same pinned weights, libraries and machine, so this is a deterministic local replay, not an external reproduction or independent-person review.

The first development audit stopped before model replay because it compared JSON snapshots byte-for-byte; the runner's serializer escaped Unicode while the source protocol did not. The decoded JSON objects were identical. Commit `39d4c4c` changed the audit to compare JSON content semantically and recorded both the audit implementation hash and the auditor snapshot used by the run. The result records and protocol were not changed. The successful audits then verified the manifest and replayed all reported metrics.

The full test container's SHA-256 includes bytes beyond each evaluation range. The protocol explicitly permits that opaque hash for source integrity. It does not mean confirmation rows were parsed early. Before the development gate, the row readers stop at the exclusive development boundary without requesting the next JSONL line.

## Interpretation and limits

The accuracy advantage repeated across two disjoint 256-prompt cohorts, but the NLL gain did not meet the frozen confirmation criterion. The development interval is not a substitute for that failed confirmation. Calibration diagnostics also have a transfer limitation: alpha is estimated from heads trained on 64 pairs per fold and applied to a head trained on all 128 pairs. The training-only OOF NLL does not validate calibration of the final head. The test result evaluates the resulting system on this dataset and model only.

This does not establish improved assistant behavior, user utility, safety, policy improvement, RLHF effectiveness, reasoning, truthfulness, generalization to other preference sources, capability gain, or frontier-scale behavior. The public dataset may overlap with model pretraining, and intervals are conditional on these three fixed training cohorts; they do not quantify training-cohort or dataset-sampling uncertainty.

## Artifacts

- [Frozen protocol](../protocols/cpu_hh_reward_model_v2.lock.json)
- [Study procedure](hh-reward-model-v2.md)
- [Development bundle](../results/cpu-hh-reward-model-v2/development/run-1/)
- [Confirmation bundle](../results/cpu-hh-reward-model-v2/confirmation/run-1/)
- [V1 non-pass report](hh-reward-model-v1-development-report.md)
