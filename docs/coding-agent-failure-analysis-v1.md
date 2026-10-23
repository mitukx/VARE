# Coding-Agent Failure Analysis v1

This analysis reads the retained raw outputs from PR #100, #101, and #102. These already-consumed tasks are debugging fixtures only; the counts below are not new validation and do not establish general coding ability.

## Failure counts

| Run | Trials | Edit applied | Public tests passed | Independent verified repairs | Primary observed failure |
| --- | ---: | ---: | ---: | ---: | --- |
| PR #100 Qwen3-4B raw unified-diff feasibility | 15 | 0 | 0 | 0 | 15 format failures before grading; two emitted patch candidates also failed to apply |
| PR #101 Transformers #49752 tools | 3 | 0 | 0 | 0 | two attempts ended with no tool call; one emitted a patch to a disallowed/wrong source location, then a corrupt patch |
| PR #102 v2 structured actions | 6 | 0 | 0 | 0 | incomplete/truncated tool calls under a 512-token turn ceiling |
| PR #102 v3 structured actions | 6 | 4 | 0 | 0 | edits were semantically wrong or incomplete; stale workspace state blocked additional edits |
| PR #102 v4 structured actions | 6 | 3 | 0 | 0 | tokenizer edits missed a required override set and repeated as no-ops; fusion attempts did not produce a valid semantic repair |

All 18 PR #102 rollouts received binary reward 0. The independent baseline/known-fix calibrations passed their expected controls, so these failures are agent attempts, not grader calibration failures.

## Representative trace diagnoses

### Tokenizer routing, v4 seed 501

The public test reported both missing override entries: `qwen3_5_moe_text` and `qwen4_exp`. The patch had already added routing rows, but the model did not make the required override update. It retried edits that returned `edit would not change the file`, reread source, and ended after inspecting the diff without a successful repair. This is a feedback interpretation and repeated-no-op failure, not an edit-format failure.

The v3 tokenizer traces made broader, incorrect changes to existing Qwen routes while adding the new map entries. Their public assertion explicitly included the missing override entries and wrong mapping, but the repair sequence did not isolate the missing state. No reference fix is encoded in the controller changes.

### Fusion registration, v4 seed 602

The first edit was rejected as stale. The model reread the file, retried with another stale revision, reread again, and then ran the public test against the unchanged baseline. The trace shows a source-revision synchronization failure followed by a failure to turn refreshed context into a valid action. In v3, seeds 401 and 402 instead applied semantically incorrect duplicate-registration logic and received the public conflict assertion; seed 402 later attempted another edit with a stale revision.

### Earlier format and tool-use failures

PR #100's 15 outputs all failed unified-diff format validation, and 2 included non-applicable patch bodies. PR #101's greedy and second sample did not call an edit tool before exhausting retries. Its first sample used a patch targeting `video_processing_minicpmv4_6.py`, which was outside the pinned editable task surface; its follow-up patch was corrupt. None of these reached an independently verified success.

## Controller changes made in this follow-up

- Structured reads are capped at 120 lines and 24,000 characters.
- Public-test graders and candidate edits use separate readable/editable allowlists; public tests can be read but not changed.
- Short per-session handles represent the exact host-side workspace revision and source preimage hashes. The editor still validates the full hashes and applies atomic preimage-checked edits.
- Tool transcripts retain full host evidence, while model-facing tool responses omit long hashes and bound public-test output to 2,000 characters.
- Repeated failures on an unchanged diff receive an explicit prompt to inspect the assertion and make a different targeted edit.
- The final diff must be inspected after the latest edit and public-test result. Finalization returns only `candidate_ready_for_independent_grading`; it always sets `verified_success` to false. The independent grader remains authoritative.
- Regression tests cover stale snapshots after reread, identical failed-test signatures, repeated no-op edits, public-test read-only access, separate transcript/model result fields, and fail-closed finalization. No task-specific reference repair is embedded.

## Limitations

The existing PR #102 runner did not save an independently versioned inference runner entrypoint in the repository; the general `scripts/run_local_agent.py` is now extended for the new opaque-handle path. This controller change alone is not an agent success. A new frozen protocol and fresh, qualified Development tasks are required before another model inference run. No Qwen3-4B retry or learning run is part of this change.
