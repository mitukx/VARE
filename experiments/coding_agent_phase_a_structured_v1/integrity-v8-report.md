# Phase A grader integrity v8

## Status

The prospective integrity calibration is **ABORTED_RESOURCE** before candidate grading. No baseline, fixed-revision, or red-team candidate was run under v8. The affected Transformers task remains a consumed Development task and is not eligible as a fresh task or held-out evaluation.

The v7 protocol, grader, adapter, and retained v7 evidence remain unchanged. V8 introduces a separate adapter and entry point. The adapter returns the parsed source registry and complete runtime registry as structured values. The trusted host reconstructs the expected values from the hash-pinned registry and optional-dependency state, then compares the complete result through the existing parent-side oracle. The candidate process no longer computes registry fingerprints, so patching its process-global `hashlib.sha256` or `json.dumps` cannot preserve a stale digest after a registry mutation.

## Frozen provenance and hypothesis

Protocol: `vare-coding-agent-phase-a-integrity-v8`

Protocol SHA-256: `24e5c48db38b2134f36b0d04c81078c49a055fe53e778722d2b8d113e49c4ab7`

Task: `transformers-49116-tokenizer-routing-v2`; retrospective, Development-only, consumed.

Upstream: `huggingface/transformers`, Apache-2.0. Baseline: `07338b6c74a578868368e6e549dea83414e4b8cb`. Known fix: `462bf5a50f6434e9610513b34537a892967ccd84`.

The falsifiable hypothesis is that returning the full AST and runtime registries to the host makes the documented fingerprint-monkeypatch attack fail the parent oracle. The v8 calibration matrix requires pinned baseline failure, exact known-fix success, and failure of a candidate that mutates a registry while replacing process-global fingerprint helpers.

The implementation lock pins the reused v7 parent grader, v8 runner, and v8 adapter. The source task manifest, expected registry, fixed container image, protocol, and regression test are also hash-identified in the lock.

## Executed verification

- `tests/test_grade_transformers_phase_a.py` and `tests/test_grade_transformers_phase_a_integrity_v8.py`: 39 passed.
- Ruff: passed.
- mypy: passed for the v8 runner and adapter.
- `git diff --check`: passed for the staged change.
- The synthetic regression changes a runtime registry entry while replacing `hashlib.sha256` and `json.dumps`; complete host-side structure comparison still rejects it.
- No Docker candidate grading, model load, model generation, or optimizer update was executed.

Raw command output and the current preflight are in `results/phase-a-integrity-v8-r1/`, with a SHA-256 manifest. The earlier `results/phase-a-integrity-v8-preflight/` snapshot records resource telemetry from the initial adapter draft; it predates the final v8 structure-comparison implementation and contains no candidate outcome.

## Resource gate

The final preflight recorded 32 GiB physical memory, 9.36 GiB available memory, 59% free memory pressure, 1.33 GiB free of 5 GiB swap, and about 125 GiB free disk. The protocol requires at least 4 GiB available memory and 2 GiB free swap before candidate grading. Available memory passed; free swap failed. The active Docker context was `default`; the retained task preflight used `colima`, so the pinned image was not inspected in this run. No model-related process was running. Cached checkpoints included Qwen3-4B-Instruct, Qwen2.5-3B-Instruct, and SmolLM2-360M-Instruct; no checkpoint was loaded.

The prior Qwen3-4B v7 attempt remains an abort before generation. This report does not rerun it or alter its threshold. No v8 grader outcome was collected, and task certification remains blocked until the resource and image-runtime gates pass and the three-call calibration completes.

## Evidence level and limits

The source-level regression reaches **E0 harness-correctness evidence** for the specific fingerprint-mutation comparison. The independent task grader has not been recalibrated under v8, so this does not certify the task, establish a real-model repair, qualify a fresh Development cohort, or authorize SFT/GRPO. The Phase A minimum eligible-task gate remains 15.

Reproduction of the bounded source checks:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest \
  tests/test_grade_transformers_phase_a.py \
  tests/test_grade_transformers_phase_a_integrity_v8.py -q
ruff check scripts/grade_transformers_phase_a_integrity_v8.py \
  benchmarks/mlsys_repo_eval_v4/transformers_49116_tokenizer_routing_v2_grader_v8/grader_adapter.py \
  tests/test_grade_transformers_phase_a_integrity_v8.py
mypy scripts/grade_transformers_phase_a_integrity_v8.py \
  benchmarks/mlsys_repo_eval_v4/transformers_49116_tokenizer_routing_v2_grader_v8/grader_adapter.py
```

Candidate grading remains disabled until a fresh preflight confirms at least 2 GiB free swap, the pinned image is available by digest, and all protocol identities match.
