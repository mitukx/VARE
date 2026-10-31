# MLSys improvement loop v1

A minimal loop in which a proposer submits a change to a real VARE code path and the trusted
parent verifies it independently before accepting. Evidence: `results/mlsys-improvement-loop-v1/`.
Protocol: `protocols/mlsys_improvement_loop_v1.json`. Evidence level: systems evidence for the
verification loop itself (scripted proposals); no learning or "self-improvement" claim.

## Workload

`parse_blocks` + `apply_blocks` in `src/vare/learning_pipeline/edit_format.py`, which turns a
policy completion into SEARCH/REPLACE edits once per rollout. It is CPU-only, deterministic, has
exact outputs (ok, reason, new text), existing tests, and a clear hot spot (`_relaxed_replace`
is O(lines × needle) with a slice per position). The module is hash-frozen for learning-pipeline
run-001, so the repository copy is never edited; candidates exist only in sandbox workspaces.

Inputs come from `benchmarks/mlsys_improvement_loop_v1/workload.py` (grader-owned): per seed,
six generated Python files of 800–2500 lines and one unique marker line per case, so no two cases
share an input. The `differential` profile adds ambiguous, missing, no-op and empty-search cases.

## Stages (fail closed, first failure is the reason)

1. **Admission**: unified diff applied with `git apply` to a scratch copy; it must modify only
   `edit_format.py` in place (no new, deleted, renamed, mode or binary entries) and change it.
2. **Policy**: AST allowlist over the whole candidate: imports from `re`, `dataclasses`,
   `functools`, `itertools`, `collections`, `bisect`, `operator`, `string`, `typing` only; no
   `exec`/`eval`/`compile`/`open`/`getattr`/`setattr`/`globals`/…; no dunder access beyond a short
   list.
3. **Static**: `static_flags_v2` from the v2 grader on added lines.
4. **Tests**: public tests (proposer-visible) plus held-out tests run through `grade_isolated_v2` in
   the attested bubblewrap sandbox; verdicts come from HMAC frames judged in the parent, with
   per-call canaries; skipped, missing or failed tests reject.
5. **Differential**: 3000 fresh edge-heavy inputs; the sandbox digest of the candidate's outputs
   must equal the digest the parent computes with the unmodified module. Expected values never enter
   the sandbox.
6. **Timing**: per rep, a fresh seed and a shuffled order of baseline, candidate and an empty
   control, each in its own fresh sandbox workspace. The parent timestamps receipt of the
   authenticated `ready` and `done` frames from `scripts/mlsys_improvement_bench_runner.py`; inputs are
   generated before the candidate is imported; output types are checked exactly so work cannot be
   deferred. Every baseline and candidate run's digest must match the parent's expectation.

Acceptance (pre-registered): all of the above, A/A calibration valid (its CI contains 1.0),
`workload_covers_startup`, speedup CI95 lower bound > 1.0 and point ≥ 1.05. Runs with the policy or
static layer disabled (audit mode) are recorded but never accepted.

## Anti-gaming summary

| attempt | stopped by |
|---|---|
| edit or add tests, conftest, other files | admission (path / structural entries) |
| skip tests (`pytest.skip`, imports of runner) | policy, static; with both off, the trusted runner reports MISSING |
| rebind the clock, memoize | policy (`import time`); with it off, parent-side clock, pre-bound runner clock and unique per-run inputs |
| special-case large or benchmark inputs | held-out tests, differential check, per-run output digests |
| rebind hashlib/json/hmac to forge the digest or frames | runner binds `_hashlib.hmac_digest`, `hashlib.sha256` and its formatting before import; frames need the per-run key from stdin |
| print fake frames | runner moves the frame channel off fd 1; unauthenticated lines carry no authority |
| lazy result objects | exact type checks (`bool`, `str`, `None`) inside the timed loop |

## Reuse

- v2 trusted grader (`learning_pipeline_isolated_grading`, `learning_pipeline_trusted_verdict`,
  `learning_pipeline_trusted_runner`): sandbox backend, attestation, frame format, static scan,
  `grade_isolated_v2`.
- `vare.isolated_candidate`: `paired_difference_ci`, `workload_covers_startup` and the timing design
  (parent clock, parent-kept expectations, per-rep seeds, randomized order with an empty control).
  Its Docker harness is not used because Docker is not required here.
- Public tests adapted from `tests/test_learning_pipeline_v1.py`.

## Using another proposer

    PYTHONPATH=src python -m scripts.mlsys_improvement_loop export-view --out VIEW
    # proposer reads VIEW/task.md, edits VIEW/edit_format.py, emits a diff
    PYTHONPATH=src python -m scripts.mlsys_improvement_loop evaluate --patch P.diff --source llm --out OUT

`evaluate` runs the attestation and A/A calibration for its session and appends to `OUT/ledger.jsonl`.

## Limitations

- Static and policy layers are pattern/AST based; the sandbox, held-out tests and parent-side checks
  are the primary controls. In-process attacks that defeat Python-level bindings (for example via
  native code) are excluded only by the import allowlist.
- Timing is on a shared host; the A/A interval is the measured noise floor for that session only.
- The held-out tests are fixed; a proposer that sees many verdicts could overfit them (the
  differential check on fresh inputs is the guard against that).
