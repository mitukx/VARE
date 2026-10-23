# SmolLM2 tool-call prompt intervention v2

**Status: `COMPLETED` diagnostic run; 0/2 independently correct repairs.** This protocol tested whether an exact first-call example would stop SmolLM2-360M from copying tool schemas as ordinary JSON. The checkpoint loaded in both matched attempts and both emitted the prescribed `list_files` call. Neither attempt read the source, edited it, or ran the public test; the independent grader returned `FAIL` for both unchanged baseline candidates.

The experiment reused MLX-LM #1910 as a Development-only task after the v1 run. It is not a fresh task cohort and provides no held-out, policy-improvement, SFT, GRPO, or learning evidence. The model, task, grader, tool definitions, runtime, seeds, sampling parameters, and budgets matched v1. The only treatment was an explicit user-message example showing the accepted call wrapper and argument keys.

| Arm | Seed | Generated tokens | Valid `list_files` | Source read | Edit applied | Public tests | Independent grade | Agent wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 31 | 130 | Yes | No | No | Not run | FAIL | 2.933 s |
| Sampled | 37 | 78 | Yes | No | No | Not run | FAIL | 1.897 s |

Across both attempts, independently correct repairs were **0/2**, unique Development tasks solved were **0/1**, valid edits were **0/2**, and public-test invocations were **0**. Binary rewards were `[0, 0]`, with mean `0` and population variance `0`. Both candidate source hashes remained the baseline hash `2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722`. No optimizer step occurred.

The treatment improved only the first tool transition: v1 made zero valid tool calls, while v2 called `list_files` in both attempts. The next transition still failed. In the greedy arm, the model supplied `{"path":{"type":"string"}}` to `read_file`, copying the parameter schema instead of supplying the string `"qwen3_coder.py"`; the runtime rejected both such calls. In the sampled arm, the model narrated the file-list result and then asked the user to provide the path. Both attempts stopped with `no_tool_call_retries_exhausted`. This is a within-task Development trace, not an independent comparison or capability estimate.

The frozen checkpoint was `mlx-community/SmolLM2-360M-Instruct`, snapshot `5d5e6fee5beb5d4d4d85e2203ef23e1230a9aac7`, with file identities inherited and rechecked against the completed load-only smoke. MLX/Metal loaded it twice. The experiment took 18.647 seconds total, including grading. Peak sampled agent RSS was 907.469 MiB (greedy) and 886.188 MiB (sampled); MLX peak allocations were 1,438,383,268 and 1,438,399,710 bytes. Host swap stayed at 3,806.5 MiB from preflight through both attempts. Minimum sampled free memory was 59%; no resource abort occurred. The public-test container remained the pinned network-disabled image.

The independently calibrated grader remained separate from candidate execution. It found the unchanged baseline in both candidate workspaces and retained the expected semantic failures for object and array inputs. Raw model outputs, tool results, grader records, source hashes, and resource telemetry are preserved under `run-1/` and covered by its SHA-256 evidence manifest. The candidate never received hidden grader assets or the known-fix patch.

Protocol SHA-256: `74c22304f4c7954b29f2589ba73edf9c88bb5260d67ab6c21f10305b7b6e6231`. Runner SHA-256: `1cfb2df04d667d642303aa69e72b7dd02f92f6be408fd70184b9610d8416a1fa`.

Reproduce the no-inference verification and the bounded run with the commands frozen in `protocol.json`. The protocol's `--check-only` path confirms task, grader, model, prompt, runtime, prior-exposure, and resource identities without generating tokens.

The result is a negative feasibility outcome for repair generation. A follow-up may test an explicit first and second tool-call example on this same Development task, but it must retain the current task exposure and cannot be used as held-out confirmation.
