# Independent AI review: urllib3 evaluator integrity CLI v1.1

## Scope and fingerprint

This is an independent AI review, not a human review. I reviewed the uncommitted generic integrity-calibration mode and protocol lock, then exercised its declared calibration contract and the task-specific production path. I did not edit code, run models, or use external compute.

- Repository HEAD: `0fd2cf6d5405a9a2e88f035e39f187695fc91b47`
- `benchmarks/historical/urllib3_3352/evaluator/grade.py`: `22b318ae35251ef1d5c9b2dd3f5f680188ad5466ec2fa7a932b28cb570b96685`
- `benchmarks/historical/urllib3_3352/protocol.lock.json`: `ac447bbfea90f92d6fb3021f1097d5d093c65358abfd411331eac76c5956b004`
- `scripts/calibrate_integrity.py`: `72442513ff320ff32e0e93a897ae39fe2189e7bceb931ba7518a63ffda279aa1`
- `scripts/grade_urllib3_3352_task.py`: `0a2bf967831598cf9212d5d993e5134e606b363fa3c25c9459016a950d9b10a6`
- `benchmarks/historical/urllib3_3352/vare_spec.json`: `74d14c28a615cf66f55e0c02ff02b73393d895728e1c190cf380a171d3779a01`
- Binary diff for `grade.py` and `protocol.lock.json`: `a3c130109a6f7d938fb46cf54a3ebfb6308a551bd8a494a5cc35b68377b2c57d`

## Integrity-only calibration

From the repository root, I ran:

```sh
<HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 scripts/calibrate_integrity.py \
  --output results/urllib3-3352-evaluation-v1/review/v1_1-integrity-cli-review-20261011/calibration
```

It exited 0 with `status=calibrated`, 4 controls, and 12/12 isolated mutations rejected across four locked tasks. For `urllib3-preserve-ambiguous-request-target-v1`, the empty-workspace control reached the expected missing-source validation (`FileNotFoundError`); mutations of `task.json`, `TASK.md`, and a copied `evaluator/grade.py` each returned `ValueError` with the expected “does not match the locked protocol” contract. The calibration's per-task records are in `calibration/controls.json` and `calibration/tamper_cases.json`; summary is in `calibration/summary.json`.

The generic mode is selected only when both `--source-record` and `--task-id` are absent. In that mode it checks the locked descriptor, brief, and evaluator, verifies candidate source files, and intentionally returns the calibration error without executing a candidate. Supplying only one of those flags is rejected by argparse. The urllib3 VARE spec supplies both flags, and the wrapper substitutes both placeholders, so the normal real-grade invocation goes through the production branch.

## Production path

Both commands were run from the repository root with the task's pinned Python 3.11 runtime:

```sh
PYTHONPATH=src <HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 scripts/grade_urllib3_3352_task.py \
  --workspace ${RUNS_ROOT}/calibration-v5/pinned-base/preparation/workspace \
  --source-record ${RUNS_ROOT}/calibration-v5/pinned-base/preparation/source-record.json \
  --task-id urllib3-preserve-ambiguous-request-target-v1 \
  --task-python <HOMEBREW_PREFIX>/bin/python3.11
```

```sh
PYTHONPATH=src <HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 scripts/grade_urllib3_3352_task.py \
  --workspace ${RUNS_ROOT}/calibration-v5/known-fix/preparation/workspace \
  --source-record ${RUNS_ROOT}/calibration-v5/known-fix/preparation/source-record.json \
  --task-id urllib3-preserve-ambiguous-request-target-v1 \
  --task-python <HOMEBREW_PREFIX>/bin/python3.11
```

Pinned base exited 1 with `passed=false`: both F2P cases failed, all three P2P cases passed, and no F2P request line was observed. Known fix exited 0 with `passed=true`: both F2P and all three P2P cases passed with the exact expected HTTP request lines. Both had `candidate_source_scope_valid=true` and no integrity errors. The raw JSON outputs are saved as `pinned-base.json` and `known-fix.json` alongside this report.

## Dispatch and lock assessment

No accidental transition to integrity-only mode or production-lock bypass was found. Both mode selectors are optional only as a pair; the integrity branch requires both absent, while the production branch requires both present and validates the task ID, source record, pinned base revision/tree, workspace baseline, and locked asset hashes. The VARE task spec explicitly supplies both required production flags. The base/known-fix runs demonstrate that this invocation reaches the production grader and preserves its fail/pass behavior.

The generic calibration is deliberately not a hostile-code sandbox and returns an error even after its input-integrity check; it does not claim task success. Its trust anchor is the unchanged protocol lock. As documented in `calibrate_integrity.py`, this check does not authenticate coordinated edits to the lock and the files it describes. Public evidence should therefore pin the repository revision and protocol-lock hash externally.

## Reproduction artifacts

- Calibration output and manifest: `calibration/`
- Production outputs: `pinned-base.json`, `known-fix.json`
- Exact review prompt: `v1_1-integrity-cli-review-prompt.txt`
- Machine-readable record: `v1_1-integrity-cli-review-record.json`

No human review, model inference, GPU work, paid API call, or code change was performed in this review.
