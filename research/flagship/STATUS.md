# VARE Flagship Mission Status

## Current checkpoint

- Last verified `origin/main`: `416f35c465712898e8e5764a8e43193ef86e7afb`; PR #159 is integrated.
- PR #150 is confirmed merged with its exact-head CI, CPU-evidence, and E2E checks passing.
- PR #160 is open at `e9a5fc42a289c5820547977f1eaa18f41f70336c`. The #64575 model-preflight integrity erratum is included; fresh exact-head CI is running and CPU evidence passed. Main requires nine check contexts and zero approving reviews.
- PR #161 is open as a draft at `f005f8da1135e875f498e31e51fc43b811aa17ad`, synchronized with #160 head `e9a5fc42`. Fresh exact-head checks are running.
- PR #162 is being synchronized with the updated #161 parent; its fresh exact-head checks must pass before integration.
- Milestone: `p15-fresh-vllm-64830-source-qualification`.

## Completed in this checkpoint

- The historical #64575 model attempt remains **NOT RUN**: the old preflight skipped task JSON/prompt hashes, although retrospective hashes match. The resource NO-GO remains valid and the original receipt/protocol are unchanged; see the append-only erratum.

- Corrected `_check_artifact_hashes` so `frozen_inputs.task` is validated before generic artifact filtering. Missing task entries, paths, hashes, files, or content mismatches now fail closed.
- Corrected bounded-attempt cleanup to send `SIGKILL` to surviving process-group members even if the group leader already exited.
- Added regressions for changed/deleted task JSON and prompt files, malformed manifests, and a SIGTERM-ignoring child. Ruff and `py_compile` passed; the focused suite passed **19/19**. Result record: `results/pr160-review-fixes-v1/run-001.json`.
- Verified the exact #64830 source identity and expected changed paths. Metadata-only review found no direct identity/family overlap at moderate confidence; public pretraining exposure remains unknown.
- Preserved baseline attempt-001 as a setup failure before collection because Seatbelt denied pytest's `/dev/null` logging write. It is not counted as a behavioral result. Addendum 3 was superseded before invocation after its output roots failed to match its profiles.
- Pushed Addendum 4 and three corrected profiles at `5d2064a7`. Each arm's sole write root matched its attempt-002 output; profiles retained default-deny/network-deny rules.
- Completed the frozen three-arm comparison: baseline failed both intended post-override assertions; the wrong mutant was rejected by both; upstream-fixed passed both. All arms collected successfully, had zero skips/errors, no timeout or RSS breach, and matched frozen inputs/commands/profiles. Full receipts and bundle checksums are under `results/mlsys-fresh-vllm-64830-source-qualification-v1/`.
- Appended source/test exposure to the research-agent ledger. There was no model exposure or inference, training, Held-Out access, paid compute, or API use.
- The modified custom-speculator behavior remains excluded because its full registration and VllmConfig validation fixtures are outside the overlay.
- Both duplicate exact-head CI runs on `8c29522fa4f48dbc0922256df75fba43adc6cef3` passed: Python 3.11/3.12/3.13, lint, typecheck, blob budget, Math500 grader, seeded-summary CPU, RVL inference CPU, and calibrated CPU tasks. This state-only checkpoint will advance the PR head and needs fresh exact-head CI before PR #162 is marked ready.

## Capability and research status

- PR #64830 is **QUALIFIED_SOURCE_LEVEL** only for the HiSparse and watermarking method-level cases. It remains **UNASSIGNED**. This does not establish agent repair, Coding RL improvement, whole-PR qualification, or pretraining independence.
- Coding capability and RL update remain **NO-GO**. No model inference, training, or Held-Out access occurred.

## Immediate next action

Finish integrating the #160 → #161 → #162 → #163 stack in order, with each exact-head check set green. Then use the already demonstrated local Qwen3-4B route for one prospective #64830 Development repair attempt under a new frozen protocol and model-specific resource gate; do not retry either closed 14B protocol.
