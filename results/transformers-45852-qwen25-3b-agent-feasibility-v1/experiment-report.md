# Qwen2.5-3B coding-agent feasibility on Transformers #45852

- Overall experiment status: **COMPLETED**.
- Learning gate: **NO_GO**.
- Training/optimizer comparison: **NOT_RUN**.
- Held-out evaluation: **NOT_RUN**.
- Task: `transformers-45852-4d-mask`, Development-only and now consumed.
- Protocol SHA-256: `2fc9a1361aad3640672290d1b1d1ba23e27098bc5808004608554ce16d465154`; protocol lock SHA-256: `74834c0526f638b4ce2ca966528f7e2a5604cb75b7f3e78836d316746bff749b`.
- Runtime code commit: `fd4803d651c3dd6e5bf7741639aba4e6e0850534`; independent grader commit: `b9f5dd6b32bacad026b87a5fed43dac2108e142a`.

The hypothesis was that at least one of two bounded Qwen2.5-3B-Instruct-4bit attempts would produce a patch accepted by all six cases in the calibrated independent grader. The result was **0/2 independently accepted patches on one task**. The binary rewards were `[0, 0]`; within-task sample variance was `0.0`. Two generations on one task do not provide a trainable task-diverse signal.

The greedy run generated 875 tokens in 44.34 seconds, used all 12 turns, applied one diff, and exhausted its turn budget without a successful finish. Its diff changed only two lines in the Apache license header. The public syntax smoke passed, but the candidate passed only 3/6 independent cases; all three Torch 4D-mask cases still failed with wrong outputs. The unaffected ordinary padded-batch and NumPy controls passed, as they do on the pinned baseline.

The sampled run generated 202 tokens in 15.20 seconds, used five turns, made no edit, and stopped after exhausting the no-tool retry allowance. It also passed only 3/6 cases, matching the baseline behavior.

Both generations first sent an extra `path` argument to `list_files`, which the frozen v1 runner's empty schema rejected. The sampled trace then supplied the literal placeholder `<allowed source path>`; the runner correctly rejected that path. The greedy trace repeatedly searched for nonexistent helper names and eventually submitted an unrelated license-header patch. The public test was intentionally syntax-only, so its pass did not measure the requested behavior.

The first process launch selected system Python and failed before loading MLX. That raw failure is preserved under `run-1/greedy/`; it loaded no model, constructed no prompt, and generated zero tokens. The two actual generations used the locked MLX environment. No prompt was exposed to the model during the failed launch.

Resource limits passed for both actual generations. Peak child RSS was 1932.8 MiB, peak MLX allocation was 2.713 GB, host free memory reached a minimum of 43%, and swap remained 3758.44 MiB with no measured growth. The two independent grader passes took 55.11 seconds total, with each candidate container limited to 1536 MiB and 30 seconds per case. No paid compute was used.

## Reproduction

Use a clean checkout at the protocol's runtime commit and the pinned baseline Transformers source. Set `MLX_PYTHON` to the Python executable in the locked MLX environment, `MODEL_SNAPSHOT` to the verified local snapshot, and `WORKSPACE` to a clean staged candidate workspace. Run each arm in a new output directory:

```bash
python3.12 scripts/monitor_local_agent_resources_v2.py \
  --output "$OUTPUT_DIR" --max-seconds 180 --swap-abort-mib "$SWAP_ABORT_MIB" \
  --memory-free-abort-percent 35 --rss-abort-mib 8192 -- \
  "$MLX_PYTHON" scripts/run_local_agent.py \
  --workspace "$WORKSPACE" --model-path "$MODEL_SNAPSHOT" \
  --model-repository mlx-community/Qwen2.5-3B-Instruct-4bit --backend mlx \
  --public-test-image localhost:5000/vare/transformers-cpu@sha256:5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f \
  --task-root results/transformers-45852-qwen25-3b-agent-feasibility-v1/task-input \
  --output "$OUTPUT_DIR/agent-run.json" --instruction "$LOCKED_INSTRUCTION" \
  --seed "$SEED" --max-turns 12 --max-no-tool-retries 1 \
  --max-new-tokens-per-turn 256 --max-total-new-tokens 2048 \
  --max-context-tokens 6144 --wall-seconds 180 --temperature "$TEMPERATURE" \
  --top-p 0.95 --top-k 20 "$SAMPLING_FLAG" --edit-format unified_diff
```

For the greedy arm, use seed 113, temperature 0.0, and `--no-do-sample`. For the sampled arm, use seed 127, temperature 0.7, and `--do-sample`. Before each arm, verify the protocol's start thresholds and set the swap abort limit to the lower of 4096 MiB and the measured starting swap plus 128 MiB. Use `DOCKER_CONTEXT=colima` for candidate tests and grading. Grade each final target using `verify_protocol()`, `verify_baseline()`, `export_baseline()`, and `grade_workspace()` from the exact grader commit in `protocol.lock.json`. The full generation transcripts, grader outcomes, resource samples, and candidate workspace archives are retained beside this report.

This is a real-model Development feasibility result plus the previously calibrated E1 task/grader evidence. It is not an E4 coding capability or learning result. The exact task/model pairing is retired from future held-out use. The source-level controller corrections prompted by the traces are recorded in [controller-followup-v1.md](controller-followup-v1.md); they did not change or rerun this experiment. Any further model feasibility work must use a new protocol and another eligible, independently trusted Development task. This task will not be rerun.
