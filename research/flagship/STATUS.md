# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing.
- PR #160 is open at `71aa74ab6a8f6f1625505b68f7f332c49fabff93`. Its Python 3.11/3.12/3.13 CI, lint, typecheck, seeded-summary, RVL inference, Math500 grader, and CPU-evidence checks pass. GitHub reports merge state BLOCKED; normal review and branch protection remain.
- PR #161 is open as a draft at `ade8a0ac3dfe609e67698638faf6eb826101dbc5`, stacked on #160. Its exact-head CI and CPU evidence pass. No review is recorded.
- Active branch: `codex/freshdev-qualification-vllm64830-oct31`; draft PR #162 is open at `bd565d12193728c971d68f778b110ac3e98ca392`, based on PR #161. CI is running after the mutant evidence checkpoint.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused suite passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified the exact #64830 source identity and expected changed paths. Metadata-only review found no direct identity/family overlap at moderate confidence; public pretraining exposure remains unknown.
- Preserved baseline attempt-001 as a setup failure before collection because Seatbelt denied pytest's `/dev/null` logging write. It is not counted as a behavioral result. Addendum 3 was superseded before invocation after its output roots failed to match its profiles.
- Pushed Addendum 4 and three corrected profiles at `5d2064a7`. Each arm's sole write root matched its attempt-002 output; profiles retained default-deny/network-deny rules.
- Completed the frozen three-arm comparison: baseline failed both intended post-override assertions; the wrong mutant was rejected by both; upstream-fixed passed both. All arms collected successfully, had zero skips/errors, no timeout or RSS breach, and matched frozen inputs/commands/profiles. Full receipts and bundle checksums are under `results/mlsys-fresh-vllm-64830-source-qualification-v1/`.
- Appended source/test exposure to the research-agent ledger. There was no model exposure or inference, training, Held-Out access, paid compute, or API use.
- The modified custom-speculator behavior remains excluded because its full registration and VllmConfig validation fixtures are outside the overlay.

## Capability and research status

- PR #64830 is **QUALIFIED_SOURCE_LEVEL** only for the HiSparse and watermarking method-level cases. It remains **UNASSIGNED**. This does not establish agent repair, Coding RL improvement, whole-PR qualification, or pretraining independence.
- Coding capability and RL update remain **NO-GO**. No model inference, training, or Held-Out access occurred.

## Immediate next action

Push the fixed-arm receipt, final source-only report, append-only exposure update, and checkpoint. Update PR #162 with all three arm results and the scope limits, then verify exact-head CI and review state. Keep #64830 unassigned; any Development attempt needs its own prospective frozen assignment and resource preflight.
