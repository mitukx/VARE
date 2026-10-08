# HH length-normalized DPO development v1

**Decision: non-pass.** This outcome-informed CPU development study found no held-out preference-agreement gain from length-normalized DPO over standard DPO. It does not measure downstream task success.

## Question and scope

After HH human-preference DPO v2 failed to beat the frozen base and a post-hoc diagnostic showed strong response-length sensitivity, this study compared a length-normalized DPO margin with standard sequence-sum DPO and chosen-only SFT. The objective was to test whether dividing the DPO relative sequence log-probability margin by the mean chosen/rejected response-target count (including EOS) changes raw sequence-sum preference accuracy.

This design was informed by prior outcomes and diagnostics. It is a development comparison, not a fresh independent confirmation.

## Frozen setup

- Protocol: [machine-readable protocol](../protocols/cpu_hh_length_normalized_dpo_development_v1.json) and [lock](../protocols/cpu_hh_length_normalized_dpo_development_v1.lock.json), committed before opening this cohort.
- Model: cached Qwen2.5-0.5B-Instruct, frozen hidden features, rank-16 adapter on the output projection; this was not a full-model update.
- Data: Anthropic HH-RLHF `helpful-base` training split only. Hash-ranked training prompts were ranks `[384,512)` and development prompts `[512,768)`, after the locked historical and private-pilot exclusions. The official test rows were not parsed. These are disjoint prompts from the same public training split; possible exposure during public-model pretraining cannot be ruled out.
- Arms: length-normalized DPO, standard DPO, and chosen-only SFT, matched at initialization and optimizer budget, three fixed seeds (`7403`, `7411`, `7419`), two epochs, CPU only.
- Primary outcome: raw sequence-sum human-preference pair accuracy on 256 development prompts. This label-agreement measure is not task success.
- Gate: frozen minimum baseline, gain and paired-bootstrap thresholds, seed consistency, SFT non-inferiority, KL ceiling, and NLL guard. No checkpoint was selected after inspecting development metrics.
- Lock SHA-256: `df3feb8abbb7eb64ae7c87f7a41c99ac9ef2ca428e05b115ef88e85b6ddf4c3e`.

## Results

| Arm | Pair accuracy | Pair NLL | Mean token KL to base | Per-seed pair accuracy |
| --- | ---: | ---: | ---: | --- |
| Frozen base | 0.5000 | 47.7998 | 0 | — |
| Standard DPO | 0.4974 | 48.3019 | 0.0442 | 0.4922 / 0.5000 / 0.5000 |
| Length-normalized DPO | 0.4974 | 48.6301 | 0.0622 | 0.4922 / 0.5000 / 0.5000 |
| Chosen-only SFT | 0.4857 | 49.8483 | 0.6709 | 0.4844 / 0.4844 / 0.4883 |

The length-only baseline scored `0.5645`, higher than every model arm. Length-normalized DPO minus base accuracy was `−0.0026` (paired prompt-bootstrap 95% interval `[−0.0143,+0.0091]`). It tied standard DPO exactly in mean pair accuracy (difference `0.0000`; paired interval `[0,0]`). The candidate was `+0.0117` versus SFT (interval `[−0.0039,+0.0299]`), which passed the frozen SFT non-inferiority rule. The candidate's KL and NLL guards passed. Its required gains over base and standard DPO and both seed-consistency gates failed, so the overall decision is non-pass.

## Execution and audit record

All nine training runs completed and all nine adapter files were saved. The frozen runner then raised a `KeyError` while assembling its aggregate summary: its metric accumulator did not contain the token-normalized accuracy later requested by the summary writer. The fixed protocol and runner snapshot were not changed. The summary was reconstructed from the retained adapters using the separately implemented offline scoring path, and the frozen auditor passed selection replay, adapter score replay, and decision replay. The audit is a same-host replay; it is not an external reproduction or a retraining audit.

The runner's exception occurred before its wall-time and peak-RSS metadata could be written. Training completed in about 36 minutes on the local CPU; no timeout or memory-guard exception occurred. Exact peak RSS is unavailable. The protocol capped wall time at 7,200 seconds and process RSS at 6 GiB. The [recovery script](../scripts/recover_cpu_hh_length_normalized_dpo_development_v1.py) rebuilds the summary from the retained adapters through the locked offline scoring implementation.

The [run bundle](../results/cpu-hh-length-normalized-dpo-v1/development/run-1/) retains protocol/code snapshots, all nine adapter states, reconstructed summary, manifest, and audit output. The formal protocol/runner passed GitHub CI before execution; the CI run does not validate this run's scientific result.

## Interpretation and limits

This result does not support length normalization as a useful improvement on this cohort. It adds a second HH human-preference DPO non-pass and does not establish model capability, downstream task improvement, general preference optimization behavior, or an external reproduction. The same public training split and the outcome-informed study history limit the strength of the evidence. Preserve the result as a negative comparison; do not open a confirmation cohort from it.
