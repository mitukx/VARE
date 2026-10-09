# CPU Qwen ticket-tool feasibility v1

## Question and protocol

Can the cached Qwen2.5-0.5B-Instruct policy complete a fresh, synthetic ticket-management task using four exact, state-dependent tool operations under a CPU-only budget? The frozen base-only screen used 48 prompts across four ordered state-transition families, one greedy response per prompt, an exact final-state oracle, and a strict JSON tool-call schema. Passing would only permit a separate update/save/reload cost smoke; it would not establish a policy improvement or capability gain.

The protocol, task generator, runner, auditor, model revision and artifact hashes were committed before inference in `da528a6`. The model was loaded offline from the pinned local snapshot. No GPU, paid API, network request, or model update was used. The full raw bundle, audit, and SHA-256 manifest are retained in [run-1](../results/cpu-qwen-ticket-tool-feasibility-v1/run-1/).

## Result

The run completed all 48 responses in **178.36 seconds**, with a peak RSS of **3,316,187,136 bytes (3.09 GiB)**. Both resource limits passed. The independently implemented same-host auditor reconstructed all tasks, replayed the authorized target calls, checked the retained hashes and provenance, and reported no integrity failures.

| Measure | Result | Frozen gate |
| --- | ---: | ---: |
| Exact final-state success | **0/48 (0%)** | 8–38 overall; 2–10 in every family |
| Wilson 95% interval | **[0%, 7.41%]** | descriptive only |
| JSON-array parse | 47/48 | descriptive |
| Schema-valid outputs | **0/48** | at least 44 |
| Matched authorized tool calls | **0/120 (0%)** | at least 90% |
| Unsafe/unauthorized mutations | 0 among 0 executable schema-valid calls | 0 maximum |
| Completed prompt/response pairs | 48/48 | 48 |
| Wall time | 178.36 s | at most 1,200 s |
| Peak RSS | 3.09 GiB | at most 22 GiB |

Each family scored 0/12. 47/48 responses parsed as JSON arrays, and none of the 48 outputs passed the frozen schema. The model commonly emitted nested objects such as `{"assign_ticket":{"ticket":"...","assignee":"..."}}`; the frozen contract required a flat `{"tool":"assign_ticket", ...}` call. Thus no returned call was executable under the declared schema, and the plans were incomplete under the exact task contract. The zero unsafe-call count must not be read as evidence of safe behavior: there were no schema-valid calls to execute.

Transformers also warned that the attention mask could not be inferred because the pad token equals the EOS token and no mask was supplied. This condition is retained as an implementation limitation. No prompt, parser, decoding, task, or threshold change was made after opening the cohort.

## Audit and interpretation

The auditor exited successfully with `failures: []`, meaning the retained bundle and frozen protocol were internally consistent. Its separate decision field was `gate_passed: false`. Integrity-audit success and task-feasibility failure are different findings. A second read-only code review was performed before inference; this was not an external human reproduction.

This is a **non-pass for the exact model / prompt / decoding / schema / task pairing**. It shows that this setup did not satisfy its required interaction contract. It does not show that Qwen cannot use tools, that it lacks the underlying state-transition reasoning, or that any RL method would fail. Because the output schema and attention-mask handling may have affected responses, the run is evidence about the frozen interface as implemented, not a clean measurement of general tool-use ability.

This feasibility screen is not a novel post-training method and contains no trained policy, matched baseline, or independent downstream capability test.

## Decision

**STOP this pairing; do not tune or rerun this cohort.** The resources were sufficient, but the base policy produced no output accepted by the frozen contract, so an update-cost smoke would have no informative successful behavior to preserve. Do not count this screen as model improvement. Given the series of failed base/task gates, the next priority is independent reproduction of substantive retained VARE evidence or a concrete, independently useful trainer defect with a baseline reproducer—not another nearby model/prompt screen.

## Reproduction and artifacts

The bundle contains the exact protocol snapshot, generated task pack, raw model responses, run metadata, and independently computed `audit.json`. Rerun the auditor against the retained bundle with:

```sh
python scripts/audit_cpu_stateful_ticket_feasibility_v1.py \
  --bundle results/cpu-qwen-ticket-tool-feasibility-v1/run-1
```

Pre-run validation included Python compilation, JSON parsing, both CLI help paths, source-hash checks, and equality of the runner-generated and independently reconstructed 48-task packs. All 120 authorized target calls replayed to their frozen expected states. The repository test suite and CI were not run for this documentation/evidence update.
