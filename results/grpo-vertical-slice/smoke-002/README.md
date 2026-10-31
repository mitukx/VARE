# GRPO vertical slice smoke-002 (new identity; exact-ratio top_p 1.0) — PARTIAL, stopped by a frozen rule

Protocol: `protocols/grpo_vertical_slice_smoke_v2.json` (sha256 `60901022…5531`),
frozen and committed before the run. Train tasks only (8 admitted v1 Train
tasks from 3 Train repos); no held-out data loaded or evaluated; no capability
claim. smoke-001 (top_p 0.95, approximate-ratio evidence) is unchanged.

**Labels.** Sampling: T 0.7, top_p 1.0, untruncated (exact-ratio objective;
`docs/grpo-exact-ratio-sampling-decision-v1.md`). Rewards: v1 native grader,
**provisional** pending trusted-grader requalification (directive D1).

## What happened

* Survey completed: 32 real policy-generated completions (base policy, LoRA
  B = 0; behavior checkpoint `grpo:ffa7233847751f14:step0`), every one graded.
  6/8 groups mixed, 1 all-zero, 1 all-one; mean reward 0.406 (13/32);
  4/32 completions hit the token limit. About 11.0 s generation and 0.7 s
  grading per completion; survey wall time 373 s. MLX peak 3.56 GiB.
* Step 0 (update on the survey's informative groups) was running when the
  frozen stop rule `swap_used_fraction_max 0.97` was breached (observed 0.974
  and 0.971; macOS grew the swap file from 8 to 14 GiB). The operator stopped
  the run (SIGTERM to this run's process only), as the protocol specifies.
  After the stop, swap use fell back from 13.3 GiB to 8.9 GiB.
* Therefore **not verified**: gradient update, loss/KL, changed parameters,
  checkpoint save/reload, post-update generation. No `steps.jsonl` or
  checkpoint exists.

Failure class: resource limit (unified memory / swap) during the
behavior-log-prob + update pass, not a correctness failure. This protocol is
consumed and is not rerun; a follow-up needs a new identity with a
memory-bounded update (for example a fixed MLX cache limit or fewer
informative groups per step), frozen before running.

## Files

`run_meta.json`, `events.jsonl`, `survey.json`, `train_tasks.json`,
`samples.jsonl` (per completion: seed, top_p, tokens, reward, outcome,
output sha256; output text removed), `operator_stop.txt`.
