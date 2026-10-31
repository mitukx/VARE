# Learning pipeline v1, run-001: results

Real (non-synthetic) run of the frozen protocol
`protocols/coding_policy_learning_mlx_4b_v1.json`
(sha256 `b004df4e66e32d235ceaec011492842271a286bec3ffb126b6af8170de53bb77`, the file committed in `5d305f78`).
Code: commit `5d305f78` (frozen code hashes in the protocol; the pipeline verifies them at start).
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, MLX on an Apple M1 Pro (32 GB).
Wall clock: 4987 s (preflight projection 8131 s; limit 10800 s). No stop rule fired.

## Verdict

**Neither arm met the preregistered success criterion.** Both comparisons are
`no_detectable_change` (Holm-adjusted exact McNemar p ≥ 0.05). This is a negative / inconclusive result,
not evidence of improvement.

## Primary outcome (held-out, n = 42, greedy, 1 sample/task, `trusted_full_v1` pass)

| arm | pass | rate | Wilson 95% CI |
|---|---:|---:|---|
| no_update | 15/42 | 0.357 | 0.230 – 0.508 |
| sft_reference | 14/42 | 0.333 | 0.210 – 0.484 |
| rft_self | 20/42 | 0.476 | 0.334 – 0.623 |

## Paired comparison vs no_update (preregistered family)

| arm | both pass | arm-only | base-only | neither | lift | Newcombe 95% CI of lift | exact McNemar p (two-sided) | Holm p | criterion met |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| sft_reference | 10 | 4 | 5 | 23 | −0.024 | −0.167 – +0.120 | 1.000 | 1.000 | no |
| rft_self | 15 | 5 | 0 | 22 | +0.119 | +0.011 – +0.221 | 0.0625 | 0.125 | no (p) |

The pipeline's own `comparison.json` gives the same counts and p-values; its paired bootstrap 95% intervals are
[−0.167, +0.119] (sft_reference) and [+0.024, +0.214] (rft_self).

Interpretation, kept deliberately narrow:

- **sft_reference**: no sign of an effect (4 vs 5 discordant pairs). It also produced more malformed/unapplied
  edits (10 vs 5) and the only cheap-verifier false positive in evaluation (cheap 15, trusted 14).
- **rft_self**: all 5 discordant pairs favour the arm and the lift (0.119) exceeds the 0.10 minimum, but with
  only 5 discordant pairs the smallest attainable two-sided exact p is 0.0625, so this design could not reach
  α = 0.05 at this discordance, and Holm p = 0.125. Treat as a hypothesis for a larger preregistered run, not a finding.
  Exploratory, not preregistered: one-sided p = 0.031, mid-p = 0.031; rft_self vs sft_reference 8 vs 2 discordant, p = 0.109.
- Held-out tasks come from only 3 repositories (14 each: python-hyper/h11, msiemens/tinydb, borntyping/python-colorlog).
  McNemar treats tasks as independent; within-repository correlation makes the effective sample smaller.
  rft_self gains: tinydb 6→9, colorlog 4→6, h11 5→5. sft_reference: tinydb 6→3, colorlog 4→6, h11 5→5.
- Single seed, single greedy sample per task, one training run per arm (100 iterations, LoRA rank 8).
  rft_self trained on only 16 self-generated examples from 12 tasks (above the 8-example / 4-task floor).

## Secondary measures

| arm | cheap pass | cheap false positive | malformed/unapplied | applied but wrong | finish=length |
|---|---:|---:|---:|---:|---:|
| no_update | 15 | 0 | 5 | 22 | 4 |
| sft_reference | 15 | 1 | 10 | 18 | 5 |
| rft_self | 20 | 0 | 5 | 17 | 4 |

Train rollouts (no_update policy, 40 tasks × 4 samples, T = 0.7): 32/160 trusted pass (Wilson 95% 0.145 – 0.269),
12/40 tasks with ≥ 1 pass, 5 tasks all-pass, 7 mixed groups (GRPO-informative), 0 cheap false positives.

Training: sft_reference 78 examples (74 train / 4 valid rows), final train loss 0.088, val loss 0.789 → 0.121, 937 s;
rft_self 16 examples (12 / 4), final train loss 0.016, val loss 0.557 → 0.254, 942 s. Both adapters have
56/56 non-zero LoRA-B tensors.

## Statistics method

Pass = `trusted_full_v1` (fail-to-pass + pass-to-pass) verifier passed. Exact McNemar = two-sided binomial test on
discordant pairs, p = min(1, 2·P(X ≤ min(b, c))), X ~ Bin(b + c, 0.5); Holm over the two preregistered comparisons.
Wilson score 95% CIs per arm; Newcombe (method 10) 95% CI for the paired difference. Recompute with
`python3 analysis/stats.py .` from this directory (stdlib only; output in `analysis/stats.json`).

## Files

Committed here: run metadata, preflight, comparison, eval summary, training report and logs, LoRA configs and
metrics, training-row provenance, admitted-task calibration list, and the per-task graded evaluation rollouts
`eval/{no_update,sft_reference,rft_self}/graded.jsonl` (schema `vare.learning_pipeline.rollout/v1`: checkpoint id/step,
task id/group, decoding parameters, output, diff, per-test results for both verifiers).

Not committed (adapters, checkpoints, training datasets, raw generations, calibration results, and the 2.7 MB train
rollout log `rollouts/graded.jsonl`): listed with sha256 and size in `SHA256SUMS.txt` (`hash_only` rows).
Final adapters (`adapters.safetensors`, 14,692,068 bytes each): sft_reference
`7e7be1eff10f4ba944f0e979ce38c13d361b5f26754611419a914d1bc1f4a442`, rft_self
`5fb794f69958558c2f346d40d2d312c68c2612883dc70b1cc81d7dcf67de7cb6`.
Train rollout log `rollouts/graded.jsonl`: `cb18ed4af65a4f542008a49b6a42d8c8ee3b848cf0775ea292e47f63a9e121b3`.
