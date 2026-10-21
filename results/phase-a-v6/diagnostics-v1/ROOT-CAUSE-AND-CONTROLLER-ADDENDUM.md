# Phase A v6 Run-2 Root-Cause Addendum

Date: 2026-10-22

This is a versioned correction to the interpretation of `protocol-v1/run-2`. It does not edit that run, its protocol, its report, or its original `SHA256SUMS`. The historical report calls the exit-125 outcomes timeouts. The raw records show that interpretation was wrong.

## Root cause

The study runner created candidate workspaces with `tempfile.mkdtemp()` and no `dir`, so macOS placed them under `/var/folders/...`. The agent passed the canonical host path `/private/var/folders/.../workspace` to Docker. The study used Colima, whose VM has the user's `/Users` home mount and does not contain `/var/folders`. The host workspace was still live while the tool ran; this was a VM mount visibility problem, not workspace cleanup or candidate test behavior. The failed Docker daemon response explicitly says `bind source path does not exist`.

The raw trace analyzer verifies the original run's SHA-256 manifest before counting outcomes. It finds **7 public-test container invocations**: 6 explicit `run_public_tests` calls plus 1 automatic post-edit call. All 7 returned Docker exit code 125 with the missing bind-source error. No test assertion ran. This corrects both the prior count of six and its timeout label.

The independent hidden grader did run after the agent process for all 12 rollouts and returned a task-level failure for each. One candidate raised `KeyError`; the other candidate outputs did not satisfy their graders. There were 0 independently verified repairs. The public-test outage deprived the model of real assertion feedback, but the separate hidden grader did not report an outage.

## Trace analysis

Across 12 rollouts, the model generated 14,802 tokens over 120 tool turns; recorded prompt-token counts sum to 341,869. The raw serialized tool-result payloads occupy 101,915 characters. Run-2 did not retain the exact model-visible JSON payload for every turn, so the character count is a raw-output measure, not a token estimate for the visible subset.

- The agent made 28 source searches and 39 source reads, covering 31 distinct path/line-range regions; 8 reads repeated an exact region. Distinct reads are not labeled semantically useful without a reliable oracle.
- Six edit tool calls occurred in 3 rollouts. One edit applied. The other 5 were rejected because both opaque handles were omitted, despite a preceding read of the target. The three edit attempts followed 4, 3, and 6 earlier tool calls respectively.
- There were 0 format parse failures, 19 no-tool retries, 3 `finish` calls, and 0 accepted finalizations.
- Stop reasons: 6 exhausted no-tool retries, 4 exhausted the turn budget, 1 exhausted context, and 1 stopped after an edit without a subsequent tool call.
- Six explicit public-test calls plus one automatic post-edit test all failed at Docker launch. The per-rollout detail and hashes are in `run-2-trace-analysis.json`.

## Changes and falsifiable hypotheses

1. **Workspace visibility hypothesis:** placing workspaces under Colima's shared home mount makes the existing isolated public-test command execute while preserving a read-only workspace mount and `--network=none`. This passed a live Colima baseline-fails/known-fix-passes fixture and the 8 baseline/fixed public calibrations for the four pinned v6 tasks. These consumed tasks are infrastructure fixtures only.
2. **Handle completion hypothesis:** host-binding omitted handles to the latest read for the exact requested path will prevent safe edits from being rejected solely because the model omitted both opaque handles. The binding still requires a prior editable-path read, verifies current workspace revision and source preimage, and rejects one-sided or stale explicit handles. Deterministic controller regressions pass; no new model rollout was run to estimate its effect.

Public test results now carry a trusted classification. Exit 125 is `container_launch_failure`; exit 137/139 is a resource failure; ordinary nonzero exits are test failures; the runner's wall-time exception is a candidate timeout. Launch, integrity, and unavailable-evaluator outcomes stop the rollout, cannot satisfy finalization, and cause the study runner to skip hidden grading and leave reward null. Model-facing launch feedback is bounded and omits the host temporary path.

Candidate workspaces are created under `~/.cache/vare/candidate-workspaces`. Each task directory is private to the host user; the mounted workspace is traversable by the container's unprivileged UID and remains read-only in the test container.

## Reproduction and evidence

No model inference was performed for this addendum.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_agent_structured_edit.py \
  tests/test_run_local_agent_structured_v2.py \
  tests/test_run_coding_agent_dev_study_v1.py \
  tests/test_phase_a_v6_colima_public_test_integration.py

python scripts/analyze_phase_a_v6_run2.py \
  --output /tmp/phase-a-v6-run2-analysis.json

python scripts/calibrate_phase_a_v6_colima_mount_v1.py \
  --output-dir /tmp/phase-a-v6-colima-calibration

DOCKER_CONTEXT=colima python scripts/calibrate_public_tests_v6_v2.py \
  --repository ~/.cache/vare/transformers-debug.git \
  --output-dir /tmp/phase-a-v6-pinned-task-calibration
```

`colima-mount-calibration-v1/` records the exact image digest and image ID, implementation hashes, baseline and known-fix outputs, and its own SHA-256 manifest. `frozen-task-public-calibration-v1/` records all 8 baseline/fixed outcomes for the 4 already-consumed tasks and has a separate SHA-256 manifest. `run-2-trace-analysis.json` was generated only after verifying the immutable historical `SHA256SUMS`.

After commit `e5e6d218a75903e10538411c5bc1cc330ad40ccf`, the Colima fixture was repeated as `colima-mount-calibration-v2/` on a clean tree; it records that commit and the final runner/controller hashes. The four-task baseline/fixed calibration was also repeated as `frozen-task-public-calibration-v2/`; all 8 outcomes again matched the expected fail/pass labels. Version 2 is the final-HEAD evidence; version 1 is retained as the initial diagnosis run.

The Phase A v6 model result remains **NO-GO** (0/12 independent successes, no sampled reward variation); SFT/GRPO remain **NO-GO**. The execution fix and deterministic calibrations establish a corrected path, not new coding-agent capability. A fresh task cohort and protocol are still required before new model inference.


## Final-head review corrections (2026-10-22)

Independent review of PR #106 identified two P1 defects. The workspace allocator had created the materializer destination before `materialize_workspace()` (which correctly refuses an existing path); it now allocates only a private parent and returns an absent leaf, and the materializer creates that leaf with the read permissions required by the frozen container user. A regression test materializes a real temporary pinned Git archive and verifies source and public-test hashes.

A rollout whose agent never invoked public tests has `public_test_result: null`; it is now classified as `not_run`, retained as an unsuccessful rollout, and still sent to the independent grader. Explicit launch, runner-integrity, and evaluator failures remain fail-closed.

On final source commit `91f16aebe29a5bd34018a4f60998021016dc9915`, the Colima calibration v3 passed: the synthetic broken baseline returned `test_failure` (exit 1), and the known fix returned `passed` (exit 0), using the exact frozen image digest. This calibration did not perform model inference and is not a coding-agent success. The targeted local suite passed 56 tests; Ruff, mypy, and `git diff --check` passed. CI and a second review of the corrected head are pending. The original v6 rollouts and earlier calibration directories remain unchanged.
