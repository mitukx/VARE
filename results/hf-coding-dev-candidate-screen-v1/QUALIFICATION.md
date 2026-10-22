# HF Coding Development Candidate Screen v1

## Decision

**Source and grader calibration passed for four development candidates. Model-inference readiness is NO-GO for this session.** The screen produced 4/4 baseline failures and 4/4 known-fix passes over 11 independently authored cases per arm. It did not run a model or update policy weights.

This is task/grader qualification evidence, not Coding-Agent success, held-out evaluation, or a learning result. The inference authorization remains false in the frozen screen protocol.

## Pinned task set and provenance

| Task | Repository revision | Target | Qualified behavior |
|---|---|---|---|
| `accelerate-4296-skip-first-batches-v1` | `b795b4838eb33bde60eb398649c4ab006874e3e9` | `src/accelerate/data_loader.py` | Preserve loader configuration through skipped-prefix reconstruction |
| `accelerate-4307-tuple-padding-v1` | `e203e1ddf33388e22f4a7985ff3103717c0f9181` | `src/accelerate/state.py` | Preserve tuple type and correct padding for uneven rank splits |
| `accelerate-4354-multi-cpu-barrier-v1` | `421af8bee5bda705d3c763f491b7a794f471818e` | `src/accelerate/state.py` | Avoid accelerator-only arguments on a mocked CPU barrier path |
| `trl-6625-entropy-gradient-v1` | `67dfbe211a07a8dd6ebc1a5af5b75152e10add8e` | `src/trl/trainer/utils.py` | Match full-matrix reference gradients for chunked entropy |

The task descriptors retain known-fix commits, target hashes, Apache-2.0 license-file hashes, test hashes, source manifests, and the source PR URLs. The task IDs did not appear in the pre-existing tracked VARE tree at the candidate branch base. That search does not prove absence of related concepts or source code in model training data. These source fixes are public upstream history; the model's exact training-data exposure is unknown.

The task set spans four bug families and two repositories. Three tasks are from Accelerate, and two edit `state.py`; these are not four independent repository samples. The tuple-split test simulates rank metadata and the barrier test mocks `torch.distributed.barrier`; neither proves real multi-process distributed behavior. The TRL check is a numerical correctness reference, not a full trainer run.

## Calibration results

- Frozen protocol: `benchmarks/hf_coding_dev_v1/protocol-v1.json`, SHA-256 `bbba41169f5cf0055ebf9c497f39c28ea5a5927d4e6c217f822fbcfc127cf393`.
- Final grader source SHA-256: `0fc3e7902bb94cda6df6032f19eb225d77f3e4c8c0004a697c836aa3afa6c9f3`.
- Runtime image: `vare/hf-coding-dev-arm64@sha256:0363cbea4db3e3739f09eaa6d402ed2328d1bf768bfead190af9ee706e663d97`, Linux ARM64, CPU-only PyTorch.
- Run 4 used the final grader, which verifies the frozen Dockerfile and dependency-lock hashes as well as the immutable image digest: baseline failed 11/11 cases; known fix passed 11/11. All eight grader calls recorded the expected image ID and verified the exact candidate inventory before execution.
- The exact public-test path in `scripts/run_local_agent.py` was also run with Colima and the same image: 4/4 baseline tests failed on the intended assertion or exception; 4/4 known-fix tests passed. There were no Docker launch, timeout, or cleanup failures in final run 2.
- Candidate grading runs only a staged `src` package and a request-only task API in a network-disabled container. Expected values and comparison logic remain in the parent process. The candidate receives no grader module or hidden cases. Candidate stdout is discarded by the existing grader boundary.
- Regression tests reject an added `sitecustomize.py`, an altered imported module, file and directory symlinks, hidden `.git`, a forged grade file, a missing baseline file, an unexpected executable, and a mode change. They also verify that infrastructure failures are not mapped to assertion failures.

Raw, per-case results are in `run-5/grades/`; final public-test transcripts are in `run-2/public-tests/`. SHA-256 manifests cover each result set. `run-2/grades/` is retained as an infrastructure-only failed attempt: the subprocess did not inherit `DOCKER_CONTEXT=colima`, so image inspection failed before grading. It is not included in calibration counts. Earlier successful calibration snapshots remain preserved in runs 1, 3, and 4; run 5 is tied to the final grader and preparer implementation hashes.

## Phase A v6 failure analysis

`results/phase-a-v6/analysis-v2/analysis-v1.md` re-aggregates all 12 frozen rollouts without changing their original evidence. There were 94 valid tool calls across 120 turns, 28 source searches (12 returned matches), 39 reads (31 unique exact regions and 8 repeated regions), 6 edit tool actions (1 applied, 5 rejected across 3 rollouts), 7 public-test attempts, 19 no-tool retries, zero accepted finishes, and zero independent successes. All seven public-test attempts returned Docker exit 125 because Colima could not see the temporary `/private/var/folders/...` bind source. They were launch failures, not timeouts or assertion failures.

PR #106 addresses the measured path defect by staging the candidate workspace under `~/.cache/vare/candidate-workspaces`, a Colima-visible home path, and retains isolation controls. The Colima-backed integration test passed against the exact baseline-fail/known-fix-pass fixture. PR #106 is still open; its head `59578747054179f51bf9279450b123bbcc937daa` has all 11 reported checks successful and an exact-head review with no findings.

## Model resource gate

The cached local candidate was `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, weights SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f` (2,263,022,417 bytes). MLX 0.32.4, MLX-LM 0.32.2, and Metal GPU availability were confirmed without loading the model.

The preflight uses the already frozen local-runner thresholds: at least 4 GiB available memory, at least 512 MiB free swap, and no more than 7 GiB swap in use. The recorded probe had 10,293,723,136 bytes available and 897,381,826 bytes free swap, but 7,692,552,765 bytes of swap in use, exceeding the limit by 176,359,997 bytes. `memory_pressure` reported 55% system-wide free memory, but that does not restore the breached swap headroom. Model loading and all model inference were therefore stopped before the first token. Raw hardware and model-file evidence is in `preflight-v1/`.

**Model rollout count: 0. Applied agent edits: 0. Independent agent successes: 0. Reward variance: not measured. SFT/GRPO: NO-GO.** These are absence-of-run facts, not a model failure. Training additionally lacks a Train cohort, 15+ unused held-out tasks, and any nonconstant authentic model reward signal.

## Reproduction

1. Use the source mirrors at the baseline and fix revisions recorded in each task's `task.json`; run `scripts/prepare_hf_coding_dev_v1.py` to create fresh baseline and known-fix workspaces. The preparer refuses to overwrite existing workspaces and checks the pinned baseline snapshot.
2. Build the runtime using `benchmarks/hf_coding_dev_v1/runtime/Dockerfile` and the frozen lock files. Verify its digest matches `protocol-v1.json`.
3. Set `DOCKER_CONTEXT=colima` and run `python scripts/grade_hf_coding_dev_v1.py --task-id <id> --workspace <path> --output <new-result.json>` for both arms. The grader refuses to overwrite output evidence.
4. Run `DOCKER_CONTEXT=colima PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_hf_coding_dev_v1.py tests/test_phase_a_v6_colima_public_test_integration.py`.

The next model study requires a fresh resource preflight within the frozen limits and an inference-specific protocol frozen before sampling. The four-task set is suitable only for a bounded development feasibility screen, not for SFT/GRPO or held-out capability claims.
