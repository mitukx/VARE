# BoolQ binary verifier-RLOO development v1

## Decision

**Frozen advancement gate: non-pass.** The run is useful as a small, audited post-training study, but it does not establish an RLOO-specific gain or authorize confirmation. The preregistered RLOO-minus-base gain threshold was +5 balanced-accuracy points; the observed gain was +2.69 points and its prompt-paired 95% interval crossed zero.

## Results

| Arm | Mean development balanced accuracy | Difference from frozen base |
| --- | ---: | ---: |
| Frozen base | 0.6378 | — |
| Scalar calibration | 0.6526 | +1.48 points |
| Contextual action SFT | 0.6378 | 0.00 points |
| Exact expected verifier reward | 0.6610 | +2.32 points |
| RLOO, group size 4 | 0.6648 | +2.69 points |

RLOO minus base was +0.02694, with a stratified paired prompt-bootstrap 95% interval of [−0.00584,+0.05982]. All five RLOO seeds were above base, and the KL and resource ceilings passed. The lower interval bound nevertheless crossed zero, and the point estimate missed the frozen effect-size requirement.

RLOO exceeded exact expected reward by only +0.00376 (95% interval [−0.01244,+0.01988]); it exceeded contextual SFT by +0.02694 (95% interval [−0.00585,+0.06025]). These intervals are development screening statistics, not confirmation. In this binary deterministic-reward setup, the expected RLOO reward gradient is exactly the expected-reward gradient, so the small RLOO/exact difference is finite-sample optimization behavior, not evidence of a distinct learning signal. Scalar calibration improved over base while contextual SFT tied it; these controls do not support a contextual-learning claim.

## Integrity and audit record

The study used the pinned local Qwen2.5-0.5B-Instruct and BoolQ revisions, CPU only, offline, with no paid service or GPU. It retained 128 training rows, 512 development rows, 500 RLOO updates, and 16,000 action groups. The 512-row confirmation reservation was prompt-hash checked only; its answers were not read, and the frozen development gate failed.

The first automatic audit stopped on a bootstrap implementation error. That failure is retained in [`failure.json`](../results/cpu-lm-boolq-verifier-rloo-development-v1/run-1/failure.json). A regression test exposed both the seed/prompt orientation error and a resampled-index error; both were fixed in commit `78b9347`. The original stored run was then replayed with the corrected auditor, without retraining. The corrected audit passed: it reconstructed the prompts and labels, exact model features, train-only normalization, all 500 RLOO updates and 16,000 action groups, optimizer checkpoints, metrics, intervals, and the frozen non-pass gate. Its report explicitly records the prior audit failure.

The run source was frozen at commit `bc42d06`; the auditor fix and test were published in `78b9347`. GitHub CI passed for both commits ([study preflight](https://github.com/mitukx/VARE/actions/runs/37818118194), [audit fix](https://github.com/mitukx/VARE/actions/runs/37819749855)). This is same-host reproducibility using one pinned local runtime, not external reproduction or independent review.

Study wall time was 213.9 seconds with 2.41 GB recorded peak RSS. Corrected audit wall time was 300.5 seconds with 2.97 GB peak RSS. Each process stayed below the frozen per-process one-hour and six-GiB ceilings. No confirmation run was started.

## Claim boundary and next step

This result supports only that the specified small binary-action experiment ran under its frozen CPU protocol and that a same-host auditor reconstructed a **non-pass**. It is not sequence-level RL, free-form generation improvement, preference alignment, broad reasoning, scale evidence, external validation, or broader downstream impact. Retire these confirmation ranks. A follow-up should begin with a materially different falsifiable question and a fresh protocol, rather than retuning this threshold or reusing this development cohort.

The raw bundle is [`run-1`](../results/cpu-lm-boolq-verifier-rloo-development-v1/run-1/); its final SHA-256 manifest covers the auditor report and preserves the first-audit failure record.
