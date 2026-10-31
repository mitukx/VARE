# Train-pool probe v1 (exploratory / diagnostic)

Protocol: `protocols/grpo_train_pool_probe_v1.json` (sha256 `6e2a95c0…5245`),
frozen and committed (ce418bde) before the run. `run_meta.json` records the
pre-rewrite SHA ada9da30 of the same commit; the history rewrite of 2026-11-01
changed only authorship and the base, and the protocol and probe-script blobs
are identical in both commits. Train tasks only; no held-out
data was loaded or evaluated; no capability claim.

**Evidence labels.**
* Sampling used temperature 0.7 with **top_p 0.95**. Under the exact-ratio
  sampling decision (`docs/grpo-exact-ratio-sampling-decision-v1.md`) these
  estimates describe the truncated distribution, not the untruncated top_p 1.0
  distribution that exact-ratio GRPO uses: treat them as diagnostic only and
  re-measure at top_p 1.0 before relying on them.
* Rewards come from the v1 native grader (`scoring.score_candidate`) and are
  **provisional** pending trusted-grader requalification (directive D1).
* Per-task estimates are G=4 samples; a single group's class is noisy
  (in the informativeness study, prior-mixed tasks replicated as mixed 15/21).

## Result

The candidate list (140 unused Train-repo rows, round-robin over the 5 Train
repos) was exhausted; no stop rule fired. Wall clock 2812 s (46.9 min), MLX
peak 3.19 GiB, swap fraction 0.81–0.92.

| | mixed | all-zero | all-one | gated out |
|---|---|---|---|---|
| existing v1 pool (56 tasks) | 10 | 37 | 9 | – |
| this probe (61 probed of 140) | 16 | 37 | 8 | 79 (57 target file too long, 20 reference blocks do not restore file, 2 prompt too long) |
| **total** | **26** | 74 | 17 | |

Goal (≥ 24 mixed under base G=4): reached, 26 — at top_p 0.95 and with
provisional rewards, so not yet a qualified pool. 27 tasks have pooled
p-hat in [0.1, 0.9] (secondary measure).

## Files

* `train_pool_v1.json` — per-task identifiers, task group, origin, patch and
  problem-statement sha256, k/n and class, pooled counts, input sha256s; no
  output text. Built by `build_pool.py`.
* `tasks.jsonl` — per-candidate gate, calibration, rewards, class, sha256s.
* `samples.jsonl` — per-rollout seed, reward, token counts, output sha256 (no text).
* `events.jsonl`, `run_meta.json`, `summary.json` — run log and metadata.
