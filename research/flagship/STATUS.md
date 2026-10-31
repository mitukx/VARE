# VARE Flagship Mission Status

Updated 2026-10-31 17:45 UTC. This is a state-only reconciliation of the files in `research/flagship/` against GitHub and the committed receipts. No experiment, model load, preflight, or grade was run for this update. Earlier status blocks are preserved in Git history; the dated decisions are in `DECISIONS.md`.

## GitHub checkpoint (verified 2026-10-31 17:45 UTC)

- Repository: `https://github.com/mitukx/VARE`.
- `main` is `6cafa37a2ccc14cd250a50e7005a71806e7d81c3`, the merge of PR #165 (learning pipeline v1) at 2026-10-31 17:35:53 UTC. The previous `main` was `da15797ca6345f73eba8249c9059db3b1a9de41e`, the merge of PR #163.
- PRs #140, #144, #145, #150, #159, #160, #161, #162, #163, and #165 are merged.
- PR #165 changed no file under `research/flagship/` or `results/mlsys-fresh-vllm-64830-*`, so PR #164 and this reconciliation merge into the new `main` without conflicts. Their exact-head CI below ran on the old base `da15797c`; fresh exact-head CI on a head that contains `6cafa37a` is still required before merge.
- PR #164 (`codex/qwen3-vllm64830-dev-v2` → `main`) is the active **draft**. At this checkpoint its head is `759ac3af1924e5a22b404def05c14839eabd1a8b` ("Record V7 Qwen3 development attempt outcome").
- Exact-head CI on `759ac3af`: 19/19 check runs succeeded; the last finished at 12:33:54 UTC. These are the 9 required contexts (blob-budget, lint, math500-grader, rvl-inference-cpu, seeded-summary-cpu, test 3.11/3.12/3.13, typecheck), each run on both push and pull_request, plus calibrated-cpu-tasks.
- The V7 launch head `eb82cb00` passed 19/19 after one rerun of the failed job: `test (3.12)` first timed out in `tests/test_trajectory_workers.py::test_independent_stages_bounded_backlog_and_slow_step_heartbeats` (1,616 passed, 111 skipped) and passed on rerun. The earlier heads `77b8a9b7`, `45df90fc` and `053b25fe` also passed 19/19.
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
| v6 | Qwen3-4B | Preflights PASSED before diagnostic-005 (15,852,896,256 bytes) and diagnostic-006 (14,244,691,968 bytes); no Development prompt under V6; its assignment was reused by V7 | — | no | closed |
| diagnostic-005 (Train) | Qwen3-4B | **FAILED_BEFORE_MODEL_LOAD**: a symlinked Hugging Face blob was outside the sandbox allowlist; exited after 1.565 s | no | no | closed |
| diagnostic-006 (Train) | Qwen3-4B | **FAILED_DURING_FIRST_GENERATION_METAL_JIT_SANDBOX_WRITE_DENIED**: the blob fix worked and the Train prompt reached the first generation call, but the sandbox denied the Metal compiler cache write (`monolithic_metal.pcm` under `com.apple.metalfe`); 0 tokens; exited after 3.432 s | yes | no | closed |
| **v7 Development** | Qwen3-4B | **FAILED_DURING_FIRST_GENERATION_METAL_JIT_SANDBOX_WRITE_DENIED**: no-model Metal smoke and both no-load preflights PASSED (last: 13,685,456,896 bytes / 12.75 GiB, 81% pressure free); the Development prompt was delivered and one generation call started, then the same Metal JIT write denial occurred before any token, despite the V7 write rules for the two observed cache directories; exited after 3.168 s | yes | **yes, once** | **closed, no retry** |

V7 receipt (`results/mlsys-fresh-vllm-64830-agent-attempt-v7/attempt-001/development-result.json`): 0 generated tokens, 0 native tool calls, 0 permitted edits, no agent-attempt record, no candidate (candidate source byte-identical to the baseline), 0 independent grader calls, no grader feedback. Sampled process-tree RSS peak 2,569,338,880 bytes and direct-child high-water 2,615,672,832 bytes, both under the 8 GiB limit. The live monitor recorded 4 samples (pressure free 80–81%, swap unchanged at 1,680,804,413 bytes) and no dynamic abort. The receipt calls this an infrastructure/sandbox failure, not a coding-capability result.

The committed V7 `agent-attempt.log` is byte-identical to the diagnostic-006 log (sha256 `96e2d8019f0e…`, 2,480 bytes). It contains only the sandbox-extension messages, the traceback and the Metal error, with the same cache path. The two bounded-attempt records differ in profile, workspace, task root, seed, timing and RSS.

Totals across all routes: 1 Development prompt delivered (V7), 0 generated tokens, 0 native tool calls, 0 source edits, 0 candidates, 0 independent grades. There was no training, no policy update, and no Held-Out access. The Development task `vllm-64830-v1-runner-unsupported-feature-checks` is now model-exposed (prompt delivered once, nothing generated). The earlier Qwen3-4B Train #1904 repair (2/2 grader cases) is still only an exposed Train result. Fresh Development repair success remains zero.

Memory snapshots recorded by PR #164: before diagnostic-006 (11:58 UTC) 13.27 GiB available, 83% pressure free; after diagnostic-006 13.71 GiB, 79%; before V7 (12:23 UTC) 12.75 GiB, 81%; after V7 (12:25 UTC) 13.69 GiB, 80%. Swap stayed at about 1.57 GiB used. No process was stopped.

## Learning pipeline v1 (PR #165, merged; separate from the vLLM #64830 routes)

- Frozen protocol `protocols/coding_policy_learning_mlx_4b_v1.json` (sha256 `b004df4e66e3…`). Run-001 was one real run, one seed, of Qwen3-4B 4-bit LoRA on MLX: Train 80 / held-out 42 SWE-smith tasks, split by repository (held-out: h11, tinydb, colorlog). It shares no task or repository with the vLLM #64830 Development task or the flagship exposure ledger.
- Held-out trusted pass (n = 42): no_update 15, sft_reference 14, rft_self 20. Both preregistered comparisons are `no_detectable_change` (exact McNemar, Holm-adjusted: sft_reference p = 1.0, rft_self p = 0.125). This is a **negative / inconclusive result**, not evidence of improvement; `go_no_go.rl_update` stays `NO_GO`.
- Offline follow-ups on the same run: the independent re-analysis re-derived every number (0 mismatches); the G1/G3 verifier-error pilot was aborted as `inconclusive`; hypothesis A (trusted-verification allocation) is `not_supported` (exploratory, not blind).
- Grader isolation audit (synthetic probes): the v1 native grader can be made to report false passes; the sandboxed grader quarantines the probed forgeries when the static scan is on, and a canary-aware forgery still passes with the scan off (residual risk). 0/123 Train and 0/106 held-out run-001 applied edits match the blocking patterns.
- Records: `results/learning-pipeline-v1/run-001/RESULTS.md`, `results/learning-pipeline-v1/grader-isolation-audit/README.md`, `docs/coding-policy-learning-v1.md`. Do not rerun run-001; any follow-up needs a new preregistered protocol.

## State reconciliation in this update

- Follow-up after the PR #165 merge: the GitHub checkpoint, `state.json` `git_refs`, and the blocker scope were updated, and the learning-pipeline v1 protocol and records were indexed by hash. No ledger or evidence file changed in this follow-up.
- `state.json` now indexes, with status, every v2–v7 protocol, diagnostics 001–006, and their preflight, sandbox and outcome receipts.
- The v1 NO_GO is restored as a historical blocker.
- Result hash references that no longer match the current tree are labelled, not rewritten: six are historical snapshots of mutable files, each tagged with the commit where its hash verifies; two point to external files outside the repository; three (the ledger, its sidecar and `errata/SHA256SUMS`) are labelled as pre-append snapshots at `759ac3af`, with the current hash noted.
- The canonical-JSON hash convention used for `development_repository_repair_campaign_v1.json` is documented.
- `git_refs.active_pr_state` now uses the schema value `open` (it was `open_draft`).
- The exposure ledger has six appended events: one per v2–v6 assignment file, and one for the V7 Development prompt delivery, backed by the v6 `exposure-assignment.json` that the V7 protocol freezes (no v7 assignment file exists). Supplement 002 and the errata manifest bind the new ledger hash. Prior events are unchanged.

## Known inconsistency (kept, not rewritten)

- `results/mlsys-fresh-vllm-64830-agent-attempt-v5/exposure-assignment.json` reuses the v4 `event_id` and lists the 14B model repository, although its `created_for_protocol` and `route_after_v4_no_go` fields name the Qwen3-4B V5 route. The file is unchanged; the ledger event records the discrepancy.

## Next action (plan, as stated by PR #164 at `759ac3af`)

Diagnose why the macOS sandbox denied Metal compiler extension/cache access despite narrowly scoped write rules, using no-model probes only. Do not retry V7. Any future inference requires a separately frozen protocol and a no-model proof of the exact first-generation Metal compiler path; fresh exact-head CI and a resource preflight remain mandatory. None of this has been done yet.
