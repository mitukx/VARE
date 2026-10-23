# Phase A Integrity Calibration v5

## Scope

This calibration addresses candidate workspace integrity and container-image provenance for the two existing Transformers Development task adapters. It does not run model inference, policy updates, or held-out evaluation. Protocols v1 through v4 and their evidence remain unchanged.

## Frozen boundary

- Protocol: `vare-coding-agent-phase-a-integrity-v5`
- Protocol SHA-256: `355b84c548e85299cc846e1da6dc4d9dda95403f16da492211d782f3ce5149ef`
- Grading image: `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Docker image configuration ID observed: `sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- The grader compares every file and directory, including modes, with a separately hashed pinned source manifest. Only the task's one editable target file may differ.
- The grading subject is reconstructed from the bytes captured during validation; arbitrary candidate workspace contents are not copied into the subject.
- Candidate test output is discarded by the isolated grader. Grader code and expected values are not mounted into the candidate container.

## Calibration outcomes

| Task | Baseline revision | Known-fix revision | Baseline | Known fix | Grading wall time |
| --- | --- | --- | --- | --- | ---: |
| Transformers #49116 tokenizer routing | `07338b6c74a578868368e6e549dea83414e4b8cb` | `462bf5a50f6434e9610513b34537a892967ccd84` | failed (`wrong_output`) | passed | 3.442 s / 3.167 s |
| Transformers #49412 fusion registration idempotence | `cb33194ad6152bd9fad6305378d92db385dd7b32` | `527725cace0b038a0ac8ab738e921f118bb2668a` | failed (`wrong_output`) | passed | 3.447 s / 3.423 s |

Runtime: Apple M1 Pro, 32 GiB unified memory, Colima 0.10.3 (`aarch64`), Docker client 29.9.1 / server 29.5.2. Each grading call used the existing isolated grader's 1,900 MiB container memory limit and 60 s timeout.

Raw JSON results are in `results/phase-a-integrity-v5/calibration/run-2/`. An initial calibration attempt accidentally put the known-fix source in a workspace named baseline; its passing output is retained under `attempt-1-invalid-label/` and excluded from the table above. The clean v5 run rebuilt distinct baseline and known-fix workspaces and produced the outcomes shown.

## Red-team and regression checks

`tests/test_grade_transformers_phase_a.py` completed with 28 passing tests. The red-team cases cover `sitecustomize.py`, modified imported source, malicious and target symlinks, hard links, forged grading output, hidden `.git`, missing files, extra executable files, extra directories, file mode changes, and an alternate digest-pinned image. Full `grade()` path regressions verify these are rejected before candidate grading is invoked. Late workspace mutation after the validation snapshot is not staged. The six related isolation/structured-agent suites also passed, for 90 tests total.

Raw pytest output is `results/phase-a-integrity-v5/red-team/pytest.stdout.txt`; the combined related-suite output is `results/phase-a-integrity-v5/red-team/related-tests.stdout.txt`. The lock pins the grader source and test source hashes.

## Historical runner provenance

The exact v2/v3 `scripts/run_local_agent.py` bytes with the locked SHA-256 were not recovered. All 12 versions reachable through local Git refs were hashed; none matched the historical lock. Inspected retained VARE task caches also contained no matching runner source. A full unreachable Git-object recovery was not completed, so this is not an exhaustive claim about every unreachable object. No hypothetical runner was reconstructed or represented as historical source. The v2/v3 reproducibility limitation remains explicit.

## Limitations and decision

- This is an integrity/calibration result, not evidence of Coding-Agent ability or model improvement.
- The task source and grader behavior are the same previously consumed Development tasks; this result does not create fresh evaluation data.
- P0 workspace and image integrity checks have passed locally on Apple Silicon with the frozen Docker image. CI must pass on the final pushed HEAD before this PR's P0 gate is considered complete.
- No new inference, SFT, GRPO, or held-out access was performed.

**P0 local calibration: PASS. Research capability milestone: NOT ESTABLISHED.**

## Reproduction

From the repository root, use `PYTHONPATH=src` and `DOCKER_CONTEXT=colima` with the frozen image reference recorded above. Recreate each workspace using `scripts/prepare_transformers_phase_a_v1.py`, apply the task's pinned fixed revision only to the known-fix workspace, and invoke `scripts/grade_transformers_phase_a_v1.py` with separate external evidence paths. Run the red-team suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_grade_transformers_phase_a.py
```
