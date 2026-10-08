# HH-RLHF v3 confirmation report

## Result

The frozen confirmation gate passed on 306 eligible prompt contexts. The cohort was opened only after the frozen development gate and its independent local replay audit passed. The practical-effect threshold (mean calibrated-minus-raw NLL ≤ −0.10 nats/pair) was explicitly informed by the already-opened v2 post-hoc diagnostic; this is an outcome-informed follow-up, not an untouched preregistered discovery.

| Measure | Confirmation result |
| --- | ---: |
| Eligible unique prompt contexts | 306 |
| Mean calibrated-minus-raw pairwise NLL | −0.1745 nats/pair |
| Prompt-paired bootstrap 95% interval | [−0.2337, −0.1178] |
| Frozen practical-effect cutoff | ≤ −0.10 nats/pair |
| Frozen gate | Pass |
| Independent local score replay | Pass |

Across the three fixed heads, mean raw NLL was 0.8454 and scaled NLL was 0.6709; the length-only baseline was 0.6931. Mean Brier score fell from 0.2747 to 0.2389. Mean pairwise accuracy was 0.5920 before and after scaling, as expected for positive scalar scaling, versus 0.5000 for the length-only baseline. Mean ten-bin ECE rose from 0.4310 to 0.4764. Thus the result supports an NLL/Brier improvement in this setting, not improvement under every calibration metric. ECE is descriptive and noisy at this cohort size.

## Reproduction and resources

The independent auditor separately reselected the frozen training and confirmation contexts, extracted features, refit heads and scalars, and replayed per-prompt scores. Audit status is `pass`. Its implementation hash is `26cf12a58dcdbea0081da340a817d4cfca299ae88ada6d5ef756115096890d5c`; the run manifest hash is `5003c90661e136ebdaad004945b44932b0577ecae3c05304cc66d4ab98491fcf`; protocol hash is `2a891c33b7c8c016d27e77fd95eeb4c4557d2cc2b3f1ebe193416ad95e86b4a4`.

The same audit was then rerun from a depth-1 clean clone of public commit `b35232bab1c284713d8e65aaa4ee7a4e94fab405`. It again returned `pass`; the generated `audit.json` was byte-identical to the committed result, and the clone remained clean. This checks that the committed repository contains the inputs and code needed for a fresh checkout on the same machine. Both replays used the same pinned model weights, tokenizer, software runtime, and host. Neither is external reproduction or a different model implementation. The original confirmation run used CPU only and no paid compute; peak RSS was 4,012,933,120 bytes. The complete [confirmation bundle](../results/cpu-hh-reward-model-v3/confirmation/run-1/) retains the summary, per-prompt results, protocol snapshots, environment, and audit. See the [reproduction guide](hh-reward-model-v3-reproduction.md) for a reviewer setup.

## Claim boundary and next work

This result supports the narrow claim that, for three fixed HH-RLHF reward heads and this one split/feature extractor, a positive scalar fitted on disjoint train contexts reduced pairwise logistic NLL on fresh contexts under the frozen threshold. It does not show improved ranking accuracy, because scalar scaling cannot change ranks. The threshold was outcome-informed, the three heads do not measure training-seed uncertainty, and same-host replay does not substitute for external review.

No downstream task success, policy update, RL gain, robust preference modeling, general reward-model quality, or frontier-scale behavior was tested. VARE still lacks an independently confirmed free-form task-success improvement and an external human reproduction. The next work should prioritize a small, fresh, executable-oracle task linked to policy outcomes and seek independent review/reproduction, rather than further tuning this consumed HH-RLHF cohort.
