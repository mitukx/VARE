# SVAMP/Gemma 2 2B MPS base gate v1

**Decision: stop before policy updates.** The base model had useful exact-answer success and fit the local MPS budget, but sampled group rewards were almost always uniform. Under standard group-relative centering, 15/16 observed groups provide zero reward-centered advantage. This gate is not a post-training result.

## Frozen protocol and cohorts

Protocol `svamp-gemma2-2b-mps-base-gate-v1` was frozen and pushed as `00ee6bc` before any Gemma output. It pinned the cached `google/gemma-2-2b-it` snapshot `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, local MPS/FP16 inference, and the same pinned SVAMP train file plus bounded Decimal parser. The independent equation audit excluded the inconsistent row `chal-680`.

The 100-row dev allocation was hash-selected from the 499 valid rows left after excluding the full v1 and v2 Qwen development allocations. It was disjoint from both previous cohorts; 399 rows remain training-only. The official 300-row test split was not downloaded or opened. No fitting, prompt adjustment, or policy update occurred.

The frozen gate required at least 13/64 exact greedy answers, 61/64 parseable answers, and 4/16 sampled groups with mixed binary rewards, as well as at most 30 minutes and 12 GiB RSS.

## Results

| Measure | Result | Gate |
|---|---:|---:|
| Greedy exact | 30/64 (46.9%; Wilson 95% CI 35.2–58.9%) | ≥13/64 |
| Greedy parseable | 62/64 (96.9%; Wilson 95% CI 89.3–99.1%) | ≥61/64 |
| Exact among parseable | 30/62 (48.4%) | Diagnostic |
| Sampled exact | 38/64 | Diagnostic |
| Correct responses per sampled group | 0 in 6 groups, 2 in 1, 4 in 9 | Diagnostic |
| Mixed-reward groups | 1/16 (6.25%; Wilson 95% CI 1.1–28.3%) | ≥4/16 |
| Wall time | 88.07 s | ≤1,800 s |
| Peak RSS | 6.60 GiB | ≤12 GiB |

The response and arithmetic gates passed. However, 15 of the 16 groups had four identical binary rewards. For a standard group-relative estimator with rewards (r_i\in\{0,1\}), centering gives (A_i=r_i-\bar r=0) for every response in each all-zero or all-one group. Normalizing by within-group standard deviation does not restore a reward gradient when all rewards are equal. The measured one mixed group is too little for this frozen gate; with only 16 groups, the Wilson interval is also wide. This is a conservative feasibility failure for the planned GRPO comparison, not proof that all sampling policies or training objectives would fail.

## Decision and limitations

Local inference was feasible on the cached 2B model: 88 seconds and 6.60 GiB RSS. The blocking evidence is the reward signal under the frozen policy and sampling rule, not a need for paid GPU capacity. Do not train or tune Gemma 2 2B/SVAMP under this protocol, and do not open the official test split. A credible GRPO study needs a fresh pairing or a separately justified exploration mechanism whose base gate demonstrates substantially more within-task outcome variation before training.

This is a base-policy feasibility result on elementary arithmetic. It does not establish post-training gains, broad reasoning capability, or GRPO failure generally. No paid compute was used. Reproduce with `python3 reproducers/run_svamp_gemma2_2b_mps_base_gate_v1.py`; the frozen protocol and raw outputs are retained in the repository.
