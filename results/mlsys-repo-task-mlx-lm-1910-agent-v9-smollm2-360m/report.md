# SmolLM2 MLX-LM #1910 feasibility v1

**Status: COMPLETED — two matched Development attempts completed.** The independent outcome was negative: **0/2 attempts passed**, and **0/1 unique Development tasks** had an accepted repair.

The frozen checkpoint `mlx-community/SmolLM2-360M-Instruct` at revision `5d5e6fee5beb5d4d4d85e2203ef23e1230a9aac7` loaded on MLX/Metal in both attempts. Greedy seed 31 and sampled seed 37 each generated 208 tokens over three turns. Both attempts returned tool schemas as ordinary JSON instead of invoking a tool, exhausted the two no-tool reminders, and stopped with `no_tool_call_retries_exhausted`. Neither attempted or applied an edit; the public test was invoked zero times. The candidate file remained byte-identical to the baseline (`2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722`). The independent grader returned `FAIL` for both candidates.

| Measure | Greedy | Sampled |
| --- | ---: | ---: |
| Independent grader | FAIL | FAIL |
| Generated tokens | 208 | 208 |
| Tool calls / edit applied | 0 / no | 0 / no |
| Public tests invoked by agent | 0 | 0 |
| Agent wall time | 3.175 s | 4.259 s |
| Process peak RSS (`ru_maxrss`) | 913,358,848 B | 941,867,008 B |
| MLX peak memory | 1,373,076,632 B | 1,464,958,198 B |
| Reward | 0 | 0 |

The binary reward mean was 0 and population variance was 0. This is a two-attempt observation on one Development task, not an estimate of task success or generalization. No optimizer update or training step occurred.

Resource limits held. Swap was 3822.5 MiB before and after both model processes, with 0 MiB growth. The host free-memory minimum sampled during inference was 65%; disk free was 130.366 GiB at preflight. The two model attempts and grades took 19.675 seconds total. Peak sampled process RSS was 895.266 MiB; the agent's own peak RSS record was 941,867,008 bytes. MLX-reported peak allocation was 1,464,958,198 bytes.

The Development grader was calibrated before this protocol: baseline FAIL, exact upstream fix PASS, and three deliberately incorrect mutants FAIL. Candidate code ran only in the pinned, network-disabled grader container. The candidate workspace remained pinned to baseline commit `d151a804ca04296e758e77503bc552515f723399`; its Git object database did not contain the known-fix commit. The current task exposure remains Development-only.

The failure points to tool-format comprehension in this 360M model/runtime pairing, not a memory barrier: raw outputs in `run-1/greedy/agent.json` and `run-1/sampled/agent.json` repeat the supplied function schemas rather than the requested `<tool_call>` invocation. The next justified Development-only intervention is to simplify the action catalog and give an exact worked tool-call example, then freeze a new protocol before any further model output. The 15-task eligibility gate for SFT/GRPO remains unmet, so this result does not authorize a policy update.

## Reproduction

The frozen protocol SHA-256 is `dd0330b99432b8cd30c57bb4d56e178b5a4085c6942b914a234611f98cbe267d`. Reproduce with the command recorded in `protocol.json`, using the same pinned Python environment, model snapshot, upstream clone, and Docker image. The run's `evidence-manifest.json` hashes every raw input copy, trajectory, grader result, and resource record; its sidecar verifies the manifest.

Protocol: `mlsys-repo-task-mlx-lm-1910-smollm2-feasibility-v1`  Implementation commit: `c7f29dc8d05295681bcb3686baf59ca70ca0c45a`  Protocol freeze commit: `ec77e142442d3fbaafac43684f2bd39faf70931d`  Run evidence files: 86  Run evidence-manifest SHA-256: `8842b83ffea402d87df6f4a45597bb589c995b1fce9c7f59ee51423c26907c6a`
