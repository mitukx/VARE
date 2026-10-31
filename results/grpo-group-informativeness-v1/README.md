# Group-informativeness study v1 (exploratory / diagnostic, Train only)

**Evidence level:** exploratory, descriptive. One run, small n, base checkpoint (no adapter),
Train tasks only, no held-out data loaded (`heldout_manifest_loaded: false`). No capability
claim and no significance claim.

- Protocol (frozen before the run): `protocols/grpo_group_informativeness_v1.json`,
  sha256 `cd14a290987f525938a6ceb4f3c1a88d2d70b64e7bc3817acccea10df719bcf6`
- Code: `scripts/grpo_group_informativeness_study.py run` at git `020ad28b`
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (base), T=0.7, MLX 0.32.2 on Apple M1 Pro 32 GB
- Budget: 96 rollouts per condition (288 total). All three conditions completed; no stop rule fired.
  Wall clock 3498 s (58.3 min; frozen limit 5100 s).

## Question

With the rollout budget held equal, how many GRPO groups have non-zero reward variance
(mixed groups, i.e. a usable advantage signal) under (a) uniform G=4, (b) G=8 on half the
tasks, (c) a transparent difficulty-aware sampler ranking Train tasks by the
posterior-predictive P(mixed | G=4) from run-001 Train rollouts?

## Results (`summary.json`)

| condition | groups | mixed groups | mixed fraction (Wilson 95%) | all-0 / all-1 groups | wasted generations | sec/rollout |
|---|---|---|---|---|---|---|
| uniform_g4 | 24 | 6 | 0.250 (0.120–0.449) | 14 / 4 | 72 / 96 | 14.84 |
| uniform_g8 | 12 | 3 | 0.250 (0.089–0.532) | 7 / 2 | 72 / 96 | 11.46 |
| prior_g4   | 24 | 7 | 0.292 (0.149–0.492) | 14 / 3 | 68 / 96 | 10.05 |

Wasted generations = generations in all-0 or all-1 groups (zero advantage under group-normalised GRPO).
sec/rollout differences mostly reflect which tasks each condition drew, not group size.

## Post-hoc breakdown (`analysis.json`, `analyze.py`)

Mixed fraction by the task's prior class (run-001, G=4):

| condition | prior mixed | prior all-0 | prior all-1 |
|---|---|---|---|
| uniform_g4 | 5/6 | 0/14 | 1/4 |
| uniform_g8 | 3/5 | 0/7 | – |
| prior_g4   | 7/10 | 0/12 | 0/2 |

- Across all conditions, tasks that were all-0 in run-001 were never mixed (0/33 groups, 21 distinct tasks).
  Tasks that were mixed in run-001 were mixed again in 15/21 groups.
- The run-001 prior is therefore highly predictive, and the bottleneck is supply: the frozen pool has
  only 10 prior-mixed Train tasks, so prior_g4 had to fill 14 of its 24 slots with prior-flat tasks.
  Its overall gain over uniform_g4 is small (7 vs 6) for that reason, plus the expected regression to the mean.
- G=8 did not raise the informative share per generation (0.25 in both). Split into two G=4 halves,
  the same G=8 generations give 5/24 = 0.21 mixed groups.
- An independent Train-only per-task pass-rate file (base, G=4, same run-001 graded rollouts,
  same source sha256) matches the frozen priors for all 40 run-001 tasks (7 mixed / 28 all-fail / 5 all-pass).

## Resources

MLX peak 3.15 GiB (active 2.11 GiB). System swap was 11.3–11.8 GiB of 13.0 GiB throughout
(fraction 0.872–0.888; owned by other apps, already high before launch). Free memory 77–87%,
available ≥ 14.6 GiB, free disk about 11 GiB, no thermal or performance warnings recorded.

## Files

`summary.json`, `run_meta.json`, `events.jsonl` (per-group resources), `samples.jsonl`
(per-rollout reward and timing; output text is not stored, only its sha256), `analysis.json`, `analyze.py`.
