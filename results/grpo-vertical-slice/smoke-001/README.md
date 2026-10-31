# GRPO vertical slice: smoke-001 (Train tasks only)

Evidence level: **real RL parameter updates on Train tasks**. This is not a
capability measurement. No held-out manifest was loaded (`heldout_manifest_loaded:
false` in `run_meta.json`), and no held-out evaluation was run.

Setup: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (pinned revision), MLX 0.32.2 /
mlx-lm 0.32.2, LoRA rank 8 on the last 8 layers, Adam lr 1e-5, clip 0.2,
on-policy (`max_policy_lag=0`), sampling T=0.7 / top-p 0.95, group size 4.
Tasks: the first 16 admitted Train tasks after the run-001 rollout window,
graded with `trusted_full_v1`.

Driver: `scripts/grpo_vertical_slice.py --tasks 16 --steps 3 --groups-per-step 2`.
Trainer: `src/vare/learning_pipeline/grpo_mlx.py`.

| quantity | value |
|---|---|
| survey groups (base policy) | 16 (mixed 3, all-fail 9, all-pass 4) |
| mixed-group fraction | 0.1875 (52 of 64 generations carried zero advantage) |
| steps applied | 2 of 3 (step 2: both resampled groups had zero variance, so no update, by design) |
| params changed on every applied step | yes (trainable sha256 differs before/after) |
| peak MLX memory | 3.67 GiB (load: 2.12 GiB) |
| sec/step (applied) | 117.2 s (step 0, 2105 tokens), 132.8 s (step 1, 741 tokens, incl. 57.7 s generate+grade) |
| update throughput | 31.5 / 18.0 trained tokens/s |
| checkpoint reload into a fresh model | ok (digest verified) |
| post-update generation | works; same Train tasks, descriptive only |
| wall clock | 1125 s |

Notes, reported as observed:

- The ratio is computed against behavior log-probs recomputed by a full forward
  pass at the sampling temperature, not against the sampler's KV-cached
  log-probs. The max |sampler - full forward| gap at T=1 was 0.148 nats on one
  sampled token, which is why the sampler values are not used. With gradient
  checkpointing on the update path, the on-policy ratio mean was 1.000003 / 0.99913,
  approx_kl (k3) 0.0037 / 0.0060, and clip fraction about 0.55%. This comes from
  recompute numerics on a 4-bit model, not from policy lag.
- Top-p truncation at sampling time is not modelled in the log-prob; the
  objective uses the untruncated tempered softmax.
- System swap grew from 4.9 to 12.4 GB during the run while the machine was
  shared. MLX peak stayed under 3.7 GiB, and no thermal or performance warning
  was recorded.
- `samples.jsonl` omits completion text (only `output_sha256` is kept). LoRA
  checkpoints (126 MB) are not committed.
