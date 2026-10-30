# VARE research and engineering checkpoint

**Updated:** 2026-10-31 (JST)

**Milestone:** P0 — repair and integrate PR #150 CI

**Decision:** GO for the bounded CI correction; NO-GO for another model screen or RL update until a fresh task/model pairing clears the repository's preflight gates.

## GitHub state

- Last verified `main`: `b1e5882cbf89a8f925bf6e5f0a2f288f2d4c8a02`
- Active branch: `codex/bounded-agent-file-read-v1`
- Active PR: [#150 — Bound coding-agent tools and grader receipts](https://github.com/mitukx/VARE/pull/150)
- Last verified PR head before this checkpoint's changes: `86abbd4df86dcebc539143de3f2f84b1207fff16`
- PR state at that head: OPEN, mergeable, not merged. Python 3.11/3.12/3.13 test jobs each reported the same three stale assertions; other checks passed.

## Work completed in this session

- Fetched GitHub refs and confirmed the PR and `main` revisions above.
- Read the failed job logs. The frozen v1/v2 tool contracts omit the runtime's later `replace_text` tool; the nonce test expected `malformed_result` where the parser correctly returns `protocol_mismatch`.
- Updated only the three test expectations. The v1/v2 JSON protocols and their SHA locks remain unchanged. Tests require the new runtime tool to remain excluded from the frozen allowlists; nonce mismatch must return no value.
- Reproduced the three failures, then ran the three affected test files successfully on Python 3.11, 3.12, and 3.13 (10 passed on each version).
- Independently checked all 2,194 file entries in the Stage 3 SHA-256 index: zero missing files and zero digest mismatches. The index's descriptive first line is not a checksum record.

## Capability and experiment state

- Stage 3: 8 model inference attempts over 3 previously exposed Development families; 0 independently verified repairs. One edit was applied, then failed all 14 hidden correctness cases.
- Fresh task families used: 0. Historical held-out PR #1979: untouched according to the exposure ledger.
- Coding RL / GRPO updates for this task cohort: NOT RUN; no task-success policy checkpoint was produced.
- Qwen3-Coder-30B local inference: BLOCKED by host-memory preflight; no weights were downloaded and no paid compute was used.
- The local full-suite attempt is not a valid CI comparison: this checkout is sparse and omits additional retained evidence; subprocess tests also resolved a host Python outside the isolated 3.11 environment. Do not change source based on that run. The post-push GitHub matrix is the authoritative full-suite result.

## Immediate next action

Run the three-file regression set plus `git diff --check`, commit the focused test corrections and this checkpoint, push to PR #150, inspect the new Python 3.11/3.12/3.13 Actions runs, review the final diff, and merge only if required checks and repository review rules permit. Verify the resulting commit on remote `main`, then update this checkpoint before moving to the next evidence-backed milestone.

## External blockers

- The new PR head has not yet been pushed, so its full GitHub CI result is pending.
- Capability work remains gated on a qualified fresh Development cohort and a feasible model/update path. Do not reuse the consumed Stage 2/3 tasks as fresh evidence or start RL from the zero-repair cohort.
