# Transformers #47468 Development qualification v2

V2 prospectively supersedes the untouched v1 bundle before any task outcome. V1 remains byte-for-byte as committed at `d8512fd9b320d881d91c4c86907bfefa7f24222c`; its status is **SUPERSEDED_BEFORE_OUTCOMES**, with no baseline, mutant, known-fix, or model result. V2 retains the same Development-only public task and source/test/fix/mutant identities. It does not qualify held-out or model-unseen evidence.

## Changes addressing independent review

- Before protocol identity freeze or execution, the runner queries the supplied Python process and verifies its resolved executable, full `sys.version`, version tuple, implementation and cache tag.
- It runs `pip freeze --all` for the host environment and `pip freeze --all --path <overlay>` for the explicit overlay, in an environment with `PYTHONPATH` unset and `PYTHONNOUSERSITE=1`; both outputs must byte-match their frozen v2 locks. Any discrepancy fails closed before a grader invocation.
- V2 pins the effective no-user-site host inventory. V1’s host inventory included `pypdf==6.19.0` from user site, which the grader execution environment disables. The v1 files are preserved unchanged; v2’s inventory is versioned separately.
- Before each grader launch, the runner creates exclusive output artifacts and fsyncs `phase_admitted` and `phase_started` events to the append-only hash-chained manifest. It records the process identity immediately after launch and fsyncs partial resource, stdout and stderr data during execution.
- `--recover-interrupted` reconstructs an immutable record from partial artifacts (or links an already-written terminal record), appends a recovery event, and never launches work. Any admitted phase permanently blocks a rerun, including after a partial interruption.

## Pre-outcome checks

The live runtime identity matched the v2 pins at preparation time: Python 3.12.12 / CPython `cpython-312`; host inventory 350 lines, overlay inventory 35 lines. The observed inventory files and their hashes are retained in `environment/` and `records/runtime-inventory-observation-v2.json`. These checks did not import or collect the upstream test.

Deterministic unit tests use fake subprocess results and temporary manifest/artifact directories. They cover exact runtime acceptance, Python-version drift, host/overlay dependency drift, interruption reconstruction, admitted-but-not-started recovery, and one-shot rerun rejection. They do not execute the baseline, mutant, exact fix, or a model.

Run only the v2 unit tests and source checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_transformers_47468_qualification_v2.py
python -m py_compile scripts/run_transformers_47468_qualification_v2.py
```

The ordinary runner invocation is a read-only preflight and must report `PREFLIGHT_VALID_NO_TEST_EXECUTED`. `--freeze-record` records protocol identity without running a test. The qualification sequence remains blocked pending independent review of this exact committed v2 head and the live network/grader boundary. Do not run `--execute` before that review. If an attempt is interrupted after admission, run `--recover-interrupted`; do not retry or change the protocol.

## Claim boundary

No task outcome or model result is included. This commit demonstrates only prospective protocol and harness corrections plus deterministic unit-test behavior; it establishes neither source/grader qualification nor a coding-agent capability result.
