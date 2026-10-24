# SmolLM2 sequential tool-call intervention v3

**Status: `COMPLETED` diagnostic run; `NO_GO` for further prompt tuning of this model/tool configuration on task #1910.** The two matched attempts produced zero valid tool calls, zero edits, zero public-test runs, and zero independently correct repairs. The grader returned `FAIL` for both unchanged baseline candidates. The task remains Development-only and is excluded from any held-out or fresh-task claim.

The protocol tested whether explicit, one-call-per-turn examples for `list_files` and `read_file`, with the concrete string path `qwen3_coder.py`, would carry the agent through source inspection. The treatment followed v2, where both seeds called `list_files` but failed to read the source. The model, task, grader, tool schemas, runtime, seeds, sampling settings, and resource limits remained fixed.

| Arm | Seed | Generated tokens | Valid tool calls | Source read | Edit applied | Public tests | Independent grade | Agent wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Greedy | 31 | 412 | 0 | No | No | Not run | FAIL | 4.182 s |
| Sampled | 37 | 657 | 0 | No | No | Not run | FAIL | 6.291 s |

Across both attempts, independently correct repairs were **0/2**, unique Development tasks solved were **0/1**, valid edits were **0/2**, public-test invocations were **0**, and recognized tool calls were **0**. Rewards were `[0, 0]`, with mean `0` and population variance `0`. Both candidate source hashes remained the baseline hash `2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722`. No optimizer step occurred.

The greedy response copied both `list_files` and `read_file` schemas as bare JSON, filling the 256-token turn limit. The sampled response claimed the task was complete in prose and repeated the claim after the no-tool reminders. Neither response was a valid call. This intervention therefore did not preserve v2's 2/2 first-call success and provides no repair capability signal.

The frozen checkpoint was `mlx-community/SmolLM2-360M-Instruct`, snapshot `5d5e6fee5beb5d4d4d85e2203ef23e1230a9aac7`; MLX/Metal loaded it twice. The run took 22.417 seconds total, including grading. Peak sampled agent RSS was 891.844 MiB (greedy) and 895.844 MiB (sampled); MLX peak allocations were 1,491,156,456 and 1,492,385,706 bytes. Swap stayed at 3,806.5 MiB throughout; minimum sampled free memory was 60%. There was no resource abort. The public-test image remained pinned and network-disabled.

Before protocol freeze, one direct verifier check failed because importing the v2 runner had patched the shared v1 verifier, and the initial v3 wrapper captured that patched function. No model was loaded and no inference occurred during that failure. The failure and correction were preserved in `preparation/validation-attempt-1.json`; the corrected strict verifier chain passed both direct validation and the committed `--check-only` run before inference.

The prior v1 and v2 raw outputs remain intact and are hash-linked by this protocol's prior-exposure record. The run retains the full generated text, tool traces, independent grade records, resource telemetry, and candidate hashes. The model never saw hidden grader assets or the known-fix patch.

Protocol SHA-256: `32b4a03d6e2189d42baa6742b06a358b09aaa5b55c86d3d9cecf720eacbd42f5`. Runner SHA-256: `b87d5ab5c38813cf130d40ca13e765a994772332939913c0bcfd4b4541920f74`. The pre-inference verifier failure record SHA-256: `9bf7620a0fa2ebf30bc679299544f53a1e87f0c5ac27394369ebae46e327bdb5`.

Reproduce the strict no-inference verification and bounded run using the commands frozen in `protocol.json`. The `--check-only` path verified the v1 and v2 exposure chain, task, grader, checkpoint, tokenizer-rendered prompt, runtime, and resource identities without generating tokens.

No further prompt variant will be run on this consumed task with this model/tool configuration. The next useful work is qualification of fresh Development tasks and independent graders; the repository's 15-task eligibility requirement remains unmet, so SFT and GRPO are not eligible.
