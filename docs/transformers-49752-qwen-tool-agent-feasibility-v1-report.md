# Transformers #49752 Qwen tool-agent feasibility v1

## Result

**NO-GO for SFT/GRPO and for a multi-task Coding-Agent capability claim.** One
new Development task was calibrated and then used for three preregistered
Qwen3-4B tool-agent attempts. The pinned baseline failed both independent
grader cases; the known upstream fix passed both. All three model attempts
left the baseline source unchanged, so independent success was 0/3 and the
observed binary reward variance was zero. This one task is now consumed for
development feasibility and is not eligible as held-out evidence.

The task is Transformers PR #49752, pinned to baseline
`6f2e80bdad560947903bb4bde8043ae95387069a` and fixed revision
`bc0e71ffef22474fc14fcaa4b399d824eebc0e50`. The target source hashes are in
the task protocol. Its evaluator preserves the upstream regression test's
semantic checks for per-visual-unit target sizes, patch counts, alternating
main/composite grids, and shared composite geometry. The baseline failed both
held-parent cases; the exact pinned fix passed both. Six container red-team
checks passed, including network denial, grader-source absence, stdout-forgery
rejection, path traversal and symlink rejection, timeout, and memory limit.
The fixed public smoke passed in a separate network-disabled container whose
candidate workspace was mounted read-only.

## Model run

The frozen [agent protocol](../benchmarks/mlsys_repo_eval_v3/transformers_49752/agent-protocol-v2.json)
used `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, MLX/MLX-LM 0.32.2, Python 3.11.15,
and `Device(gpu, 0)` on an Apple M1 Pro MacBook Pro with 32 GiB unified memory.
No paid API or external GPU was used. Each rollout started from a fresh copy of
the pinned baseline and used the same task, tools, 8-turn cap, 512-token
per-turn limit, 4,096 total generated-token cap, 12,288-token context cap, and
600-second wall limit. The three attempts were one greedy and two sampled
rollouts; their exact seeds, prompt, source, model and tool code hashes are in
the protocol and raw records.

| Attempt | Wall time | Generated tokens | Mean decode rate | Patch applied | Public test | Independent grader |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Greedy | 191.94 s | 1,637 | 8.53 tokens/s | No | Not reached | 0/2 |
| Sample 1 | 174.57 s | 596 | 3.41 tokens/s | No; two invalid patches | Not reached | 0/2 |
| Sample 2 | 182.40 s | 1,637 | 8.97 tokens/s | No | Not reached | 0/2 |

Two attempts repeatedly exhausted the response limit while emitting a tool
call. The other produced two patch tool calls, both rejected: one was a no-op
patch and the other was corrupt. The agent correctly used file-list, source
read, and search tools, but never reached the independent grader with a changed
source. This is an edit/patch execution failure, not evidence that the model
understood or solved the bug. The first public test feedback was not reached.

Per-attempt MLX peak memory was 4.20–4.22 GB and active memory after generation
was 2.26 GB. Host process peak RSS was 1.83–2.64 GB. System-wide free memory
reported 69% before and 62% after the three runs; `vm_stat` swap counters are
cumulative host values and cannot be attributed to these attempts. The
reported decode rates divide generated tokens by the runner's post-load
elapsed time and exclude initial model load. No optimizer update was run and
no optimizer-cost estimate was measured.

## Evidence and reproduction

- [Frozen grader protocol v6](../benchmarks/mlsys_repo_eval_v3/transformers_49752/protocol-v6.json)
- [Post-run source/calibration lock v7](../benchmarks/mlsys_repo_eval_v3/transformers_49752/protocol-v7.json)
- [Current security calibration protocol v8](../benchmarks/mlsys_repo_eval_v3/transformers_49752/protocol-v8.json)
- [Frozen three-attempt agent protocol](../benchmarks/mlsys_repo_eval_v3/transformers_49752/agent-protocol-v2.json)
- [Raw agent transcripts](../results/transformers-49752-qwen-tool-feasibility-v1/)
- [Independent grader outcomes](../results/transformers-49752-qwen-tool-feasibility-v1/grading/)
- [Baseline/fix public-test calibration](../results/transformers-49752-public-smoke-v1/run-4/public-smoke-calibration.json)
- [Agent-run calibration](../results/transformers-49752-isolated-v6/run-1/calibration.json)
- [Post-run calibration](../results/transformers-49752-isolated-v7/run-1/calibration.json)
- [Expanded subprocess red-team calibration](../results/transformers-49752-isolated-v8/run-1/calibration.json)
- [Commit-bound terminal-status calibration record](../results/transformers-49752-isolated-v11/run-2/calibration.json)

After the agent run, protocol v7 was frozen after a lint cleanup and a
Development-split error-message correction. Its separate calibration again
passed baseline-fail, known-fix-pass, source-swap rejection, and all red-team
checks. The agent rollouts remain bound to grader protocol v6 as recorded in
their preregistration; v7 is the post-agent implementation calibration before
the expanded security regression was added.
Protocol v8 adds an isolated child-process probe and baseline hash/index-flag
regressions. The updated calibration again passed baseline-fail, known-fix-pass,
source-swap rejection, and the expanded red-team suite. The process can read
only the candidate subject, with no Git history or grader source.
The v8 record retained `status: running` in an aggregate field despite storing
all terminal results. Protocol v11 corrects that evidence bookkeeping without
changing the grader semantics; its fresh baseline/fix/red-team calibration
records a terminal `complete` or `failed` status consistently in result and
progress evidence, including failed calibration paths.

Run the protocol's agent command from the VARE repository with its pinned MLX
environment, cached model snapshot, and `DOCKER_CONTEXT=colima`. The raw agent
records contain the complete prompts, tool calls, outputs, patch hashes,
timings, and resource values. Reproduction requires the local pinned model
snapshot and the digest-pinned candidate image; inference runs offline.

Calibration history is retained: v1 was interrupted before a complete record;
v2 found the missing `regex` dependency; v3 separated the candidate-visible
invocation shim from trusted expected values; v4–v6 corrected and froze the
candidate-visible smoke test, read-only test mount, and Development split
contract. Superseded attempts are not accepted evidence.

## Next gate

Do not start SFT or GRPO. First fix the bounded interface failure with a new
protocol and fresh Development tasks: either establish that the existing
agent's patch tool and per-turn budget can produce a complete diff without
post-processing old outputs, or stop if a fresh task still yields no applied
patch. At least two fresh Development tasks must independently pass before
calling Phase A successful. The previously frozen Development and Held-out
cohorts remain unchanged and were not accessed by this run.
