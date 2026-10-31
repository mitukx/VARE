# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing.
- PR #160 is open at `71aa74ab6a8f6f1625505b68f7f332c49fabff93`. Its Python 3.11/3.12/3.13 CI, lint, typecheck, seeded-summary, RVL inference, Math500 grader, and CPU-evidence checks now pass. GitHub reports merge state BLOCKED; the normal review and branch-protection gate remains.
- PR #161 is open as a draft at `ade8a0ac3dfe609e67698638faf6eb826101dbc5`, stacked on #160. Its exact-head CI and CPU evidence pass. No review is recorded.
- Active branch: `codex/freshdev-qualification-vllm64830-oct31`; draft PR #162 is open at `28a198a5d6abf1cf4d881efddc65d597d4ba41cd`, based on PR #161. Its checks are still running.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused suite passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified the exact #64830 source identity and expected three-file changes. The metadata-only overlap review found no direct identity/family overlap at moderate confidence; pretraining exposure remains unknown.
- Froze and pushed the scoped source qualification and its execution commands before test execution. Baseline attempt-001 reached pytest setup but stopped before test collection because pytest's logging plugin tried to open `/dev/null`, which Seatbelt denied. The attempt and classification are preserved; no candidate assertions executed and no later arm was run.
- Corrected the runner to direct pytest's internal log to the same arm-specific writable output directory. Added a regression test; Ruff, `py_compile`, and the test (**1 passed**) succeeded. A new addendum binds exact corrected commands, hashes, unchanged sandbox boundaries, and an attempt-002 sequence; it must be committed and pushed before the corrected baseline runs.
- The modified custom-speculator behavior remains excluded because its full registration and VllmConfig validation fixtures are outside the overlay.

## Capability and research status

- PR #64830 is **UNQUALIFIED / UNASSIGNED**. Attempt-001 is an environmental setup failure, not a baseline behavior result. The corrected baseline has not yet run.
- No model inference, training, or Held-Out access has occurred in this source qualification. Coding capability and RL update remain **NO-GO**.

## Immediate next action

Push the runner correction, its regression, attempt-001 receipt/classification, and execution addendum 3. Verify the frozen addendum is on PR #162, validate every stored command/hash, and run baseline attempt-002 exactly once. Proceed to wrong-mutant and upstream-fixed only through the addendum's gates; stop on any unexpected result.
