# Phase A Grader Integrity Calibration v6

## Scope and frozen inputs

This version follows v5 and leaves protocols v1–v5, their task splits, and historical evidence unchanged. It closes the three current-head review findings about grader-source provenance, preserving unrelated tokenizer routes, and rejecting incompatible fusion mappings.

- Protocol: `vare-coding-agent-phase-a-integrity-v6`
- Protocol SHA-256: `380231605ece1ae7ffdeb63e199dc89507a24bbbee33bea3652a794cccaa588c`
- Source commit containing the grader, adapters, lock, and expected registry: `4532216`
- Grader implementation SHA-256: `71598902440ae09bfaa4d8edf2b44f5668d77a14d4db61c41f83962738a76da7`
- Grading image: `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Observed image configuration ID: `sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Runtime: Apple M1 Pro / 32 GiB unified memory, Colima 0.10.3, arm64, Docker client 29.9.1 / server 29.5.2.

The grader now rejects a running source file whose SHA-256 differs from the v6 lock and records that hash in every result. The tokenizer task hashes a parent-held expectation covering the complete 266-entry registry AST and all 50 override entries from the pinned known-fix source, while the isolated runtime probe continues to check task routes. The fusion probe checks same-registration idempotence, then submits a distinct incompatible transform with a colliding source pattern and requires rejection without changing the mapping.

## Calibration and adversarial outcomes

| Check | Candidate source hash | Result | Wall time |
| --- | --- | --- | ---: |
| Tokenizer pinned baseline (`07338b6c`) | `831ae31e…` | rejected: `wrong_output` | 6.510 s |
| Tokenizer pinned known fix (`462bf5a5`) | `89d30109…` | passed | 6.547 s |
| Tokenizer known fix with unrelated `bart` route altered | `db48a1a0…` | rejected: `wrong_output` | 6.530 s |
| Fusion pinned baseline (`cb33194a`) | `9a05ab8c…` | rejected: `wrong_output` | 6.977 s |
| Fusion pinned known fix (`527725ca`) | `af053f2c…` | passed | 6.982 s |
| Fusion known fix that discards existing mappings and accepts a conflicting transform | `73b96eda…` | rejected: `wrong_output` | 6.999 s |

Raw grader JSON is under `results/phase-a-integrity-v6/calibration/` and `results/phase-a-integrity-v6/red-team/`. Every result binds protocol, implementation, adapter, complete baseline manifest, exact image reference and inspected image ID. The tokenizer expectation artifact is separately SHA-256 pinned by the protocol lock.

The candidate workspaces were rebuilt from the exact upstream revisions, normalized to the frozen Git file and directory modes, and checked against the complete source manifests. Only each task's allowed target differed for known-fix and adversarial candidates. The task grader and expected outputs are still run through the existing process-isolated `grade_calls` boundary; this calibration does not claim a general-purpose sandbox for arbitrary repositories.

The focused workspace-tampering suite covers startup hooks, imported-module modifications, symlinks, hard links, forged result files, hidden Git state, missing/extra paths, altered modes, and digest-pinned alternate images. It also verifies that the grader source lock rejects a modified implementation. The final focused grader/isolation regression command was:

```bash
DOCKER_CONTEXT=colima PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_grade_transformers_phase_a.py \
  tests/test_isolated_candidate.py \
  tests/test_isolated_candidate_expected_exception.py \
  tests/test_run_local_agent_tool_format.py
```

Ruff and mypy passed for the changed grader and adapters. CI must pass on the final pushed PR head before the P0 gate is closed.

## Historical protocol limitation

The exact historical v2/v3 `scripts/run_local_agent.py` source bytes were not recovered. The inspected local Git refs and retained task caches had no source matching the historical locks. Unreachable Git objects were not exhaustively searched. No source was reconstructed or represented as authentic historical runner code; the limitation remains explicit.

## Decision and limits

**Grader integrity calibration: PASS locally. Coding-Agent ability: NOT ESTABLISHED. Model feasibility: NOT RUN. SFT/GRPO: NO-GO.**

This uses two already-consumed Development tasks and is a grader-semantics/security calibration only. It creates no fresh evaluation cohort and provides no evidence of new model-generated repairs, reward variation, policy improvement, held-out capability change, or asynchronous learner quality. No inference, training update, or held-out access occurred.

To reproduce calibration, reconstruct the frozen source workspaces with `scripts/prepare_transformers_phase_a_v1.py`, apply only the pinned known-fix target file or the documented adversarial mutation, then invoke `scripts/grade_transformers_phase_a_v1.py` with the image reference above and an evidence output path outside the workspace. The current v6 raw JSON and SHA-256 manifest are the authoritative observed run records.
