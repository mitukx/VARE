# SVAMP/Qwen2.5-0.5B MPS base gate v2

**Decision: stop this model/task pairing before policy updates.** The fresh numeric-only prompt improved parse compliance substantially, but exact success and sampled reward diversity remained below the frozen thresholds. No SFT/RL update was run, and no capability-improvement claim is supported.

## Frozen design and provenance

Protocol `svamp-qwen25-05b-mps-base-gate-v2` was committed as `edf7b4b` before model outputs. It pinned Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, SVAMP revision `5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef`, and the same independently executed Decimal answer comparison as v1. V2 selected fresh task IDs after excluding v1's full 100-row development allocation. The official 300-row test split was never accessed.

The protocol required at least 13/64 exact greedy answers, 61/64 parseable greedy answers, and 4/16 sampled groups with both reward values, subject to 30 minutes and 12 GiB peak RSS. It permitted one numeric-only prompt gate on fresh IDs; a failure retires this pairing without another prompt/parser iteration.

The first runner invocation stopped before model loading because its v1 exclusion accidentally used the v2 hash namespace. That error and the correction are retained in [`preflight-1.json`](../results/svamp-qwen25-05b-mps-base-gate-v2/preflight-1.json); no model output was generated in that attempt. The corrected cohort audit confirmed 100 v2 dev rows, 499 untouched training-only rows, and zero overlap with v1 scored IDs. Correction was committed as `daa3973` before the successful evaluation.

## Results

| Measure | Result | Frozen gate |
|---|---:|---:|
| Greedy parseable | 59/64 (92.2%; Wilson 95% CI 83.0–96.6%) | ≥61/64 |
| Greedy exact | 6/64 (9.4%; Wilson 95% CI 4.4–19.0%) | ≥13/64 |
| Exact among parseable | 6/59 (10.2%; Wilson 95% CI 4.7–20.5%) | Diagnostic |
| Sampled parseable | 57/64 | Diagnostic |
| Sampled exact | 1/64 | Diagnostic |
| Mixed-reward groups | 1/16 (6.25%; Wilson 95% CI 0.28–8.33%) | ≥4/16 |
| Wall time | 31.81 s | ≤1,800 s |
| Peak RSS | 2.19 GiB | ≤12 GiB |

The simpler prompt passed the runtime gate and nearly passed parser validity, but the model still produced 53 parseable incorrect answers. Five outputs included units or other text the frozen parser intentionally rejects; their semantic correctness is unknown. The 6/64 exact rate counts unparseable outputs as failures under the frozen exact-success metric, while the 6/59 conditional rate describes only outputs the evaluator could score. Neither metric is a broad reasoning estimate. The sampled arm's 1/16 mixed-reward rate indicates inadequate observed reward variation for the planned group-relative update at this base policy.

## Interpretation and stopping decision

The failure is not a lack of local inference capacity: the base gate completed on MPS at 2.19 GiB RSS. The blocker for this pairing is the measured starting policy signal: low exact success and nearly constant sampled rewards. Proceeding to GRPO would spend effort optimizing a weak, sparse reward signal without a defensible feasibility case. Stop Qwen2.5-0.5B/SVAMP here; do not train on or retune against these scored rows. The untouched official test split remains sealed.

This is a negative base-feasibility result, not a post-training comparison, a model capability claim, or a general conclusion about RLVR. A new study needs a fresh model/task pairing with a stronger independently graded base policy and enough within-prompt reward variation, followed by a separately frozen and measured local update path. No paid compute was used.

Reproduce with `python3 reproducers/run_svamp_qwen25_05b_mps_base_gate_v2.py`. The frozen protocol, runner, preflight record, predictions, and summary are retained in the repository.
