# Behavior log-prob path diagnostic (Train tasks, 4B MLX bf16)

There are two Train tasks with one T=0.7 sample each. The per-token log-probs of the same
tokens, under the same parameters, were compared across computation paths
(`ratio_diag.py`, run from the repo root):

| path vs plain forward (A) | max abs diff (nats) |
|---|---|
| A again (determinism) | 0 |
| B: plain forward under mlx-lm grad checkpointing | 0 |
| C: inside `nn.value_and_grad` over the LoRA model | 0.132 / 0.062 |
| D: C under grad checkpointing | identical to C (0 vs C) |
| D repeated | 0 vs D |
| E: `mx.value_and_grad` w.r.t. a scalar only | 0.295 / 0.132 vs C |

The gap is introduced by the differentiated (traced) computation, not by
checkpointing. That traced path is deterministic. So `compute_behavior_logprobs`
now evaluates the same traced loss the update differentiates, and a lag-0 ratio is
exactly 1. Each step reports `max_abs_log_ratio`. Before this change, smoke-001
showed approx_kl of 0.004–0.006 at lag 0. Cost: one extra backward pass per trained
sample, about 1.4x a plain forward in this diagnostic.
