# Qwen2.5–DeepMath v7: failure-mode analysis

## Research question

In the fixed v7 base-policy generation run, how often did the 1,024-token completion cap coincide with an invalid terminal answer, and what can this single run establish about reward sparsity? This is a post-hoc diagnosis of one frozen run, not a test of changing the cap.

## Provenance and integrity

- Protocol: `qwen25_deepmath_grpo_math500_v7`; the v7 lock and its recorded FAIL decision are unchanged.
- Frozen runbook checkout: `0c2ca5953aa3dc2b21fa9d6aa3049ae680ff1ea6`; runner checkout: `95316981b8d654a60bb5d22a18bf9e5e8c4cc26e`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; weights SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Data: `trl-lib/DeepMath-103K`, revision `066c50a88d4e14cefc056e31111db2dba17f6c68`, TRAIN parquet SHA-256 `e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7`; fixed 32-prompt fingerprint `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`.
- Sampling: 4 completions per prompt, seed rule `20261012 + prompt_index`, temperature 0.7, top-p 0.95, maximum 1,024 new tokens, no system prompt. Optimizer updates: zero; MATH-500 was not loaded.
- Environment recorded by the run: free Colab Tesla T4, 15,637,086,208 bytes VRAM, CUDA runtime 12.8, driver 580.82.07, Python 3.12.13, PyTorch 2.11.0+cu128, TRL 1.1.0. Other pinned packages are retained in `summary.json`.
- Raw evidence remains in the [v7 Drive artifact folder](https://drive.google.com/drive/folders/1OokyHj2G2JW4bjVkKF4CdyLRt4zsGcA1). The result JSON SHA-256 is `7b41f9a9e5e92847269fe7eb63ce55110469146163bdf8dadfd1c5d64a7243fe`; terminal journal SHA-256 is `646ead42f473dd207f56163fbffef74cd0cd1b00cad655796eead0bd88e6e9e7`; durable partial JSONL SHA-256 is `f45513c90c87d253babd07b27f4ad8f967213dcc8808554831d37265b3606d51`.

The result JSON, terminal journal, and all 32 durable JSONL groups have matching run identities and completion records. Recounting actual token IDs through the first EOS (inclusive) and excluding any later padding gives **90,294 tokens**, matching both recorded counters. The 32 prompt IDs reproduce the locked fingerprint. Completion, reward, checker, EOS, truncation, final-box, and saved aggregate metrics were cross-checked. Raw inputs were not changed.

The result JSON reports 1,353.7284445762634 seconds, while the terminal journal reports 1,353.8884239196777 seconds, a 0.1599793434-second difference. `GateJournal.finalize` writes the result JSON before the terminal state, so the terminal state includes finalization time. Both values are retained; the journal value is used for cumulative-budget accounting. Throughput from the terminal value is 66.692 tokens/s. This timing discrepancy does not affect the frozen decision.

## Observed results

| Measure | Recount | Frozen criterion / interpretation |
|---|---:|---|
| Independent task success | 10/128 (7.81%) | Within the 5–90% band |
| Valid final-box | 101/128 (78.91%) | **Below 90%; FAIL** |
| EOS | 102/128 (79.69%) | 101/102 EOS outputs had a valid final box (99.02%) |
| Token-cap truncation | 26/128 (20.31%) | 0/26 had a valid final box |
| Mixed-reward prompt groups | 9/32 | Meets frozen minimum 6 |
| Mean population reward variance within groups | 0.0546875 | Positive |
| Training reward / independent checker disagreements | 0/128 | No disagreement observed |
| Generated tokens | 90,294 | Under 131,072 |
| Cumulative wall time | 1,353.888 s | Under 7,200 s |
| Peak allocated / reserved VRAM | 1.131 / 1.271 GB | Under the frozen cap |

There were 27 format failures. Twenty-six (96.30%) were truncated; one ended at EOS with an invalid final box. Mean length was 705.42 tokens, median 715.5, interquartile range 525.25–924, and range 153–1,024. Reward-1 completions averaged 551.7 tokens (n=10); reward-0 completions averaged 718.45 (n=118). This is a descriptive association over clustered outputs, not an independent-sample comparison.

## Manual review of all truncated completions

The classification criteria were fixed before applying labels: **A** sound progress cut off; **B** repetitive/degenerate; **C** inefficient enumeration; **D** reasoning or interpretation breakdown; **E** answer attempted but box incomplete; **F** insufficient evidence. Each row is one completion; index and completion index are zero-based. Prompt hashes are shortened to 12 hex characters. Cues below are compact readable renderings; the CSV contains the exact source substring checked by the script, plus tags and ambiguity notes. Confidence is an uncalibrated reviewer judgment, not an inter-rater score.

| Prompt index | Prompt ID prefix | Completion | Category | Reward / correct | Confidence | Short exact cue |
|---:|---|---:|:---:|:---:|---:|---|
| 0 | `78c290e3ea3e` | 3 | D | 0 / no | .99 | “the sum y will be slightly less than 1” |
| 3 | `34c7b4a7c448` | 3 | D | 0 / no | .99 | “Since B₀ is a constant” |
| 4 | `2cf918f1f38b` | 0 | D | 0 / no | .98 | “Gauss's Law states that the flux…” |
| 10 | `6a7430a58567` | 1 | C | 0 / no | .97 | “check the next multiple of 9” |
| 10 | `6a7430a58567` | 2 | C | 0 / no | .98 | “check the next multiple of 9” |
| 11 | `cf7c9f42113b` | 1 | D | 0 / no | .99 | “Dividing both sides by 2” |
| 14 | `dcfc016de0e2` | 2 | D | 0 / no | .99 | “Case 9: x = 4” |
| 15 | `eaa70772ca87` | 3 | D | 0 / no | .99 | “a^(43/2) = 1” |
| 17 | `218f6b954253` | 0 | D | 0 / no | .99 | “48x(x − 1) = 0” |
| 17 | `218f6b954253` | 3 | D | 0 / no | .99 | “A(0, 3) and B(8, −3)” |
| 19 | `903db4919f7e` | 0 | B | 0 / no | .94 | Repeats the `E[V²]` moment integral |
| 19 | `903db4919f7e` | 1 | B | 0 / no | .92 | “First, we integrate with respect to u” |
| 19 | `903db4919f7e` | 2 | B | 0 / no | .93 | Repeats equivalent moment integrals |
| 19 | `903db4919f7e` | 3 | B | 0 / no | .99 | “Var(U) = Var(X) + Var(Y)” |
| 21 | `59e82d24f374` | 2 | E | 0 / no | .99 | Begins an unfinished range box |
| 24 | `970e8c691297` | 0 | D | 0 / no | .99 | “must be homogeneous of degree 0” |
| 24 | `970e8c691297` | 1 | D | 0 / no | .98 | “c₁² − 8c₁ + 28 = 0” |
| 24 | `970e8c691297` | 2 | D | 0 / no | .99 | Restates the boundary condition as `u_x` |
| 25 | `9f62de007628` | 0 | D | 0 / no | .99 | Assumes “n is a positive integer” in an invented ansatz |
| 25 | `9f62de007628` | 2 | D | 0 / no | .99 | Uses “dx/dz = 1/z” |
| 26 | `6a65e6beae21` | 0 | D | 0 / no | .96 | “Fifth iteration (i = 4)” |
| 26 | `6a65e6beae21` | 1 | D | 0 / no | .99 | Introduces a backward Euler value |
| 26 | `6a65e6beae21` | 2 | D | 0 / no | .98 | “For the next step” with inconsistent recurrence |
| 26 | `6a65e6beae21` | 3 | D | 0 / no | .99 | Replaces the stated start with `y₀ = 0` |
| 28 | `e8279b3efced` | 0 | D | 0 / no | .99 | Claims even power retains eigenvalue −1 |
| 29 | `2461343cb23e` | 2 | B | 0 / no | .99 | Repeats the same residue expression |

Primary-label counts: **D 18, B 5, C 2, E 1, A 0, F 0**. “D” means a material defect was visible in the retained reasoning; it does not assert that every token before truncation was useless. For example, the four outputs for prompt 19 are labeled B because each spends the available budget on repeated/incomplete moment calculations; D is retained as a secondary tag where a mathematical error is also apparent. No case met A's requirement of materially sound progress toward a correct terminal answer. No output was labeled F because each retained ending supported at least one defensible primary category. The prompts themselves are not stored in the raw result, so prompt interpretation was cross-checked only against the completion's restatement and the fixed prompt ID.

## Truncation and reward sparsity

The group-level reward distribution was 23 groups with 0/4 correct, 8 with 1/4, and 1 with 2/4; hence 9/32 groups were mixed. Fifteen of 32 groups had at least one truncated completion, including 2 groups whose four completions were all truncated.

| Group type | Groups | Groups with any truncation | Truncated completions |
|---|---:|---:|---:|
| All zero reward | 23 | 12 (52.17%) | 22 |
| Mixed reward | 9 | 3 (33.33%) | 4 |
| All one reward | 0 | 0 | 0 |

All 26 truncated outputs had zero reward, but most incorrect outputs (92/118) ended with EOS. Truncation therefore accounts for 22/118 incorrect outputs; it cannot explain the overall concentration of zero-reward completions by itself. The group pattern is also not a monotonic relation: more zero-reward groups had truncation, but mixed groups had fewer groups with truncation than zero-reward groups. With 32 prompts and no randomized cap, difficulty and response behavior are confounded with truncation.

Within the nine mixed groups, correct completions were shorter than incorrect siblings in 8 groups and longer in 1. The mean within-group difference (correct minus incorrect) was −91.54 tokens; the median was −111. This conditions on only nine groups and is exploratory. It does not imply shorter reasoning causes correctness.

## Counterfactual format bound

The observed valid-format count is 101/128. Meeting the frozen 90% threshold requires at least `ceil(0.90 × 128) = 116`, a gap of 15 outputs. If only the 26 truncated outputs could change, at least 15/26 (57.69%) would have to become valid. If every truncated output became valid and all other outcomes stayed fixed, the arithmetic upper bound would be 127/128 (99.22%), because one EOS output is already invalid.

This is a **counterfactual feasibility bound**, not a forecast or measured treatment effect. The run does not show that longer generation would make any truncated output valid, correct, or reward-positive. It also says nothing about how GRPO would affect format or task success.

## Threats to validity and claim boundary

This analysis was selected after the v7 outcome was known, uses one model and one 32-prompt TRAIN cohort, and manually labels truncations without a second adjudicator. The 128 generations are grouped under 32 prompts and are not treated as IID observations. The cap was not randomized, so no causal effect of token budget can be identified. All task/reward values come from this one frozen evaluation setup; no policy update or held-out capability evaluation occurred.

Supported observations are limited to: this run produced nonzero reward variance in 9/32 groups; 26 outputs hit the cap and none had a valid final box; 101/102 EOS outputs had a valid final box; and 10/128 outputs passed the independent checker. The data do **not** establish that truncation is the main cause of sparse reward, that increasing the limit improves answer quality, that GRPO improves format or mathematics, or that these patterns generalize.

## Implications for RLVR and conclusion

Finite generation budget and reward availability co-occur in this cohort, but the diagnosis is mixed: some cap-limited outputs show repeated/incomplete work and many show visible reasoning breakdown. Most incorrect responses nevertheless terminated before the cap. The exploratory hypothesis that terminal-answer reachability constrains effective reward is compatible with these observations, but not strongly identified as the mechanism behind sparse group rewards. This artifact is a reproducible failure analysis, not a novel RLVR method or a capability result. Its scientific novelty is limited; its value is forensic clarity and a testable, bounded observation.

**Decision: Archive.** Preserve the failed gate and this post-hoc diagnostic together. Do not extend the same cohort into a performance claim. Any future causal test would need a separately approved protocol, fresh prompts, and a randomized or otherwise controlled generation-budget condition.

## Reproduction

The source raw files are intentionally not duplicated in GitHub. Download the result JSON, state journal, and partial JSONL from the retained Drive artifact folder; verify their SHA-256 values above. With Python 3.12 and the repository checkout, the analyzer uses only the Python standard library plus the checked-in prompt-ID canonicalizer:

```bash
python scripts/analyze_qwen25_v7_failure_modes.py \
  --result /path/to/vare-gpu-base-gate-v7.json \
  --journal /path/to/vare-gpu-base-gate-v7.state.json \
  --partial /path/to/vare-gpu-base-gate-v7.partial.jsonl
```

The command writes `results/math-grpo-v7-analysis/summary.json`, `truncated-failure-annotations.csv`, and `group-metrics.csv`. It exits on hash, identity, group membership, token recount, aggregate, or manual-quote mismatch. The focused CPU regression is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_qwen25_v7_failure_analysis.py
```

The environment variable avoids unrelated globally installed pytest plugins; the test file itself uses pytest only.
