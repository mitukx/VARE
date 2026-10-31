# VARE Flagship Mission Status

Updated 2026-10-31 12:10 UTC. This is a state-only reconciliation of the files in `research/flagship/` against GitHub and the committed receipts. No experiment, model load, preflight, or grade was run for this update. Earlier status blocks are preserved in Git history; the dated decisions are in `DECISIONS.md`.

## GitHub checkpoint (verified 2026-10-31 12:00 UTC)

- Repository: `https://github.com/mitukx/VARE`.
- `main` is `da15797ca6345f73eba8249c9059db3b1a9de41e`, the merge of PR #163.
- PRs #140, #144, #145, #150, #159, #160, #161, #162, and #163 are merged.
- PR #164 (`codex/qwen3-vllm64830-dev-v2` → `main`) is the active **draft**. At this checkpoint its head is `77b8a9b717c1a2f607d65f61babe8d40090d9b6a`.
- Exact-head CI on `77b8a9b7`: 19/19 check runs succeeded. These are the 9 required contexts (blob-budget, lint, math500-grader, rvl-inference-cpu, seeded-summary-cpu, test 3.11/3.12/3.13, typecheck), each run on both push and pull_request, plus calibrated-cpu-tasks. The last run finished at 11:58:27 UTC. The earlier heads `053b25fe` and `45df90fc` also passed 19/19.
- This reconciliation is stacked on PR #164 and changes only state files plus an append to the exposure ledger.

## vLLM #64830 outcomes (one Development task; verified from committed receipts)

| Protocol / diagnostic | Model | Outcome | Weights loaded | Development prompt | Retry |
|---|---|---|---|---|---|
| v1 | Qwen2.5-Coder-14B | Launch preflight **NO_GO** at 07:05 UTC (11,035,787,264 bytes available, below 12 GiB) | no | no | closed |
| v2 | Qwen3-4B | Launch preflights PASSED (14.34 GiB at 09:48 UTC, 13.00 GiB at 10:08 UTC); diagnostic-001 preflight PASSED (13.43 GiB); consumed by diagnostic-001; Development attempt NOT RUN | — | no | closed |
| diagnostic-001 (Train #1904) | Qwen3-4B | **FAILED_BEFORE_FIRST_GENERATION**: the sandbox denied `/dev/null` for the pinned Git revision check; exited after 3.149 s | yes | no | **must not be retried** |
| diagnostic-002 | 14B | Frozen, never run | no | no | superseded |
| v3 | Qwen2.5-Coder-14B | Preflight **NO_GO**: stale frozen bounded-runner and preflight-runner hashes (12.28 GiB available, about 0.29 GiB margin) | no | no | closed |
| diagnostic-003 | 14B | Frozen, never run | no | no | superseded by the V4 NO_GO |
| v4 | Qwen2.5-Coder-14B | Preflight script PASSED, but the overall route is **NO_GO**: the extra 2 GiB margin fell short by 1,362,055,641 bytes | no | no | closed |
| v5 | Qwen3-4B | Preflight PASSED (13.56 GiB) | — | no | closed |
| diagnostic-004 (Train) | Qwen3-4B | **SANDBOX_RUNTIME_IMPORT_FAILURE**: `mlx_lm.load` could not be imported inside the sandbox; exited after 0.539 s | no | no | closed |
| v6 | Qwen3-4B | Preflight before diagnostic-005 PASSED (15,852,896,256 bytes available); the one assigned Development attempt is unused | — | no | active |
| diagnostic-005 (Train) | Qwen3-4B | **FAILED_BEFORE_MODEL_LOAD**: a symlinked Hugging Face blob was outside the sandbox allowlist; exited after 1.565 s | no | no | closed |
| diagnostic-006 (Train) | Qwen3-4B | Frozen at `77b8a9b7`. A no-model probe read one byte from each canonical blob and confirmed the hidden grader directory is denied. **NOT RUN** | no | no | one attempt only |

Totals across all routes: 0 Development prompts delivered, 0 generated tokens, 0 native tool calls, 0 source edits, 0 independent grades. There was no training, no policy update, and no Held-Out access. The earlier Qwen3-4B Train #1904 repair (2/2 grader cases) is still only an exposed Train result; it is not evidence of Development generalization. Fresh Development repair success remains zero.

The resource snapshots recorded by PR #164 around diagnostic-005 (11:43–11:45 UTC) were 13.9–14.8 GiB available, 81–83% memory-pressure free, and about 1.57 GiB swap used. No process was stopped.

## State reconciliation in this update

- `state.json` now indexes, with status, every v2–v6 protocol, diagnostics 001–006, and their preflight and outcome receipts.
- The v1 NO_GO is restored as a historical blocker.
- Eight result hash references that no longer match the current tree are labelled. Six are historical snapshots, each tagged with the commit where its hash verifies; two point to external files outside the repository. No hash was changed.
- The canonical-JSON hash convention used for `development_repository_repair_campaign_v1.json` is documented.
- `git_refs.active_pr_state` now uses the schema value `open`.
- The exposure ledger has five appended events, one per v2–v6 assignment file, backed by `exposure-assignment.json` and the receipts. Supplement 002 and the errata manifest bind the new ledger hash.

## Known inconsistency (kept, not rewritten)

- `results/mlsys-fresh-vllm-64830-agent-attempt-v5/exposure-assignment.json` reuses the v4 `event_id` and lists the 14B model repository, although its `created_for_protocol` and `route_after_v4_no_go` fields name the Qwen3-4B V5 route. The file is unchanged; the ledger event records the discrepancy.

## Next action (plan, as stated by PR #164 at `77b8a9b7`)

Diagnostic-006 may run only after this head's exact-head CI and a fresh V6 resource preflight pass. The exact-head CI has now passed; the fresh V6 preflight has not been recorded. If diagnostic-006 loads the model, parses a native tool call, and applies a nonempty permitted Train edit, a fresh immediate V6 preflight may authorize the one assigned Development attempt. Stop on failure; there is no retry under diagnostic-006.
