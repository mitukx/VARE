# Real-agent interface v1: execution report

## Result

**NO-GO for model learning or an interface-improvement claim.** The frozen four-attempt Development comparison stopped after three starts when the pre-registered swap-growth abort fired. Two attempts completed independent grading; neither applied a source edit, ran public tests, or passed the hidden grader. The third was aborted before an agent record or grading. The final paired attempt was not started.

## Frozen design and qualified tasks

The study used Qwen2.5-Coder-7B-Instruct-4bit at revision `019cc73c45c770444708a6dd8690c66243cc5c80`, greedily decoded, against two newly qualified upstream MLX-LM Development repairs: PR #1980 (pipeline shards with split quantization metadata) and PR #1978 (RotatingKVCache mask after wrap). The planned arms were the existing unified-diff tool and bounded `read_file_range` plus exact unique-string replacement. Each task/arm cell had one attempt, seed 101, 12 turns, 768 generated tokens per turn, 4096 total generated tokens, 32768 context, four public-test calls maximum, 120-second public-test timeout, and 900-second outer wall cap.

Before inference, six Docker calibrations passed their expected controls: both pinned baselines and both wrong mutants failed public and trusted tests; each pinned upstream repair passed both. Candidate and trusted grading ran in separate containers and workspaces using the same pinned image; hidden tests were mounted only in the trusted grader. The frozen E0 probes passed against that image. Model file hashes and task/grader hashes are pinned in the protocol.

## Measurements

| Attempt | Outcome | Source edit | Public tests | Independent hidden grade | Wall time |
|---|---|---:|---:|---:|---:|
| #1980, structured replacement | No-tool retries exhausted after boundary-only ChatML outputs | No | 0 | Fail | 53.6 s |
| #1978, unified diff | Context budget exhausted after repeated full-file reads | No | 0 | Fail | 381.4 s |
| #1980, unified diff | Aborted: swap grew 123.2 MiB, over the 32 MiB cap | No | Not run | Not run | 443.4 s |
| #1978, structured replacement | Not started after the resource abort | — | — | — | — |

The complete graded set is 0/2 verified repairs. These two observations are not a valid arm comparison: they cover different tasks, and the planned pair was incomplete. In the old interface attempt, repeated full-file reads grew the prompt to 32,751 tokens against the 32,768-token context limit. The structured attempt stayed within a much smaller prompt on a different task but still failed to produce an edit, so this does not establish that the interface change caused a performance gain.

During attempt 3, swap use increased by 129,171,456 bytes (123.2 MiB), exceeding the 33,554,432-byte cap. The candidate workspace remained unchanged at the pinned baseline hash. The process was stopped before an agent record or grader call; its partial generated token stream was not persisted by the current runner. Resource samples, stderr/stdout files, materialization identity, and the empty-diff audit are retained.

## Decision

No eligible positive repair trajectory exists, and the 15-task learning gate is unmet. Do not run SFT or GRPO from this data. This execution stopped when its frozen swap-growth abort fired; the two completed attempts also produced no valid edits. The incomplete run does not establish broader model-resource feasibility. Stop this protocol here. No held-out capability, statistical improvement, asynchronous execution, recovery, or RSI claim is supported.

The exact protocol, task qualifications, six raw calibration records, wrong-mutant patches, E0 evidence, two complete raw trajectories, one aborted resource trace, and setup history are included in this result tree. Model weights are not included; the pinned revision and eight SHA-256 values are recorded in the protocol.
