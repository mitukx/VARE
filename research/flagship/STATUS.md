# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing.
- PR #160 is open at `71aa74ab6a8f6f1625505b68f7f332c49fabff93`. Its Python 3.11/3.12/3.13 CI, lint, typecheck, seeded-summary, RVL inference, Math500 grader, and CPU-evidence checks pass. GitHub reports merge state BLOCKED; normal review and branch protection remain.
- PR #161 is open as a draft at `ade8a0ac3dfe609e67698638faf6eb826101dbc5`, stacked on #160. Its exact-head CI and CPU evidence pass. No review is recorded.
- Active branch: `codex/freshdev-qualification-vllm64830-oct31`; draft PR #162 is open at `a1ecc14a384931c3af9f642b33aef1ea60a0549b`, based on PR #161. CI is running after the baseline evidence checkpoint.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused suite passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified exact #64830 source identity and expected changed paths. Metadata-only review found no direct identity/family overlap at moderate confidence; pretraining exposure remains unknown.
- Preserved baseline attempt-001 as a setup failure before collection because Seatbelt denied pytest's `/dev/null` logging write. No candidate assertions executed.
- Pushed runner logging correction and regression in `ba25adfe`; Ruff, `py_compile`, and its regression (**1 passed**) succeeded. Addendum 3 was preserved but superseded before execution after its output roots failed to match its inherited profiles.
- Pushed Addendum 4 and three corrected profiles at `5d2064a7`; each profile's sole write root matched its arm's attempt-002 output and retained default-deny/network-deny rules.
- Baseline attempt-002 passed: both HiSparse and watermarking cases failed at the intended post-override assertion (line 73), with successful collection, zero skips/errors, no timeout/RSS breach.
- Wrong-mutant attempt-002 passed its rejection gate: the same two cases failed at line 73; collection succeeded, zero skips/errors, no timeout/RSS breach. Full receipts and checksums are under `results/mlsys-fresh-vllm-64830-source-qualification-v1/wrong-mutant-attempt-002/`.
- The modified custom-speculator behavior remains excluded because its full registration and VllmConfig validation fixtures are outside the overlay.

## Capability and research status

- PR #64830 is **UNQUALIFIED / UNASSIGNED** pending upstream-fixed and final isolation gates. Passing baseline and mutant gates alone is insufficient.
- No model inference, training, or Held-Out access has occurred. Coding capability and RL update remain **NO-GO**.

## Immediate next action

Run the exact stored upstream-fixed attempt-002 `runner_argv` once from Addendum 4. Require both cases to pass with no skips, errors, timeout, resource breach, or isolation issue. Then produce the source-only qualification report and update PR #162 with exact-head CI and review status.
