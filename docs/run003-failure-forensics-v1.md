# Run-003 failure-forensics layer

Status: the forward-only classification code and public synthetic fixture matrix are implemented and exercised on the target Apple Silicon Mac. This is diagnostic-harness evidence, not a new R7 validation or a change to Run-002.

## What the retained R7 evidence says

Run-002 attempted 903 tasks. It retained 103 valid reference outcomes, 769 `reference_tests_missing`, and 31 `reference_fails`; the frozen 26-repository by 8-task floor admitted only 12 repositories, so the freeze correctly refused. The refusal and report remain unchanged.

The frozen `RepoEnv.grade` runs selected pytest node IDs, parses short-summary lines from stdout, and records any unreported requested node as `MISSING`. `run002_build_check.py` maps any reference result with a nonempty missing list to the single historical reason `reference_tests_missing`. It does not preserve the collection report, raw stdout/stderr, or per-node statuses for these invalid attempts. The reason therefore conflates at least potentially different outcomes, and the retained per-task aggregates are insufficient to recover their distribution. All 769 stay historically `reference_tests_missing`; this diagnostic layer does not recode them.

One separate, sanitized `python-dotenv` diagnostic in the Run-002 record observed pytest exit 4, 121 requested IDs unreported, and a missing-module category. It is outside the 903 attempts and does not establish a cause for the other 768 such cases. Its raw output and IDs were not retained, so a dependency repair cannot be safely specified from that artifact alone.

## Implemented diagnostic contract

[`scripts/run003_failure_forensics.py`](../scripts/run003_failure_forensics.py) accepts supervisor-owned counts and integrity/timeout signals, parses bounded JUnit XML into counts plus a digest, and emits no test names or traceback text. It distinguishes environment construction failure, collection/import failure, no tests collected, requested node absent, executed failure, pass, timeout, sandbox violation, integrity error, and unknown. Exit code 4 alone is deliberately `unknown`; an absent-node classification requires exact parent-side evidence. The old reason code is retained as a separate field.

The layer cannot yet obtain trustworthy exact node-ID inventories from arbitrary repository pytest sessions. The public matrix supplies exact missing-node metadata because its fixture is deliberately known. A future repository diagnostic runner must add an isolated collector and source/dependency/sandbox manifests before this field can be populated for target data.

## Public fixture run

The reproducible run is [`results/run003-failure-forensics-public/report.json`](../results/run003-failure-forensics-public/report.json). It used six public synthetic fixtures on macOS arm64, CPython 3.12.12, pytest 8.4.2. All six classified as intended: one pass, one executed test failure, one import/collection failure, one no-tests-collected result, one absent-node result, and one supervisor timeout. Total process wall time was 1.303 seconds (median 0.207 seconds). Raw JUnit XML and captured output were discarded; the report retains only counts, hashes, and timings. No Run-002 repository, task, or held-out data was accessed. This validates the diagnostic distinctions and report hygiene only.

Reproduce with:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.12 -m pytest -q tests/test_run003_failure_forensics.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.12 scripts/run003_fixture_diagnostics.py
```

The test-only `PYTEST_DISABLE_PLUGIN_AUTOLOAD` setting is necessary in this local system Python environment, where an unrelated globally installed pytest plugin stack fails to load. The Run-002 repository venvs are separate; this local failure is not evidence about the 769 R7 cases.

## Root-cause and Run-003 status

No per-attempt R7 collection artifacts were retained, and no immutable dependency lock exists for the old extra-dependency mapping. The only supportable distribution for the 769 cases is therefore “unclassified from retained evidence.” No dependency, test-selection, or qualification correction was made. The original 26 × 8 threshold is unchanged; no target repositories were rerun, no model-facing data was created, and the Run-002 NO-GO remains authoritative.

[`protocols/run003_failure_forensics_v1.json`](../protocols/run003_failure_forensics_v1.json) freezes this diagnostic-only scope and lists the prerequisites for any target validation: evaluator hashes, a complete dependency lock, a pinned sandbox identity, a trustworthy collection record, a prospectively fixed sample, and unchanged repository/task gates. A future target run must use a new protocol instance and preserve the exposure separation. This protocol does not authorize a held-out learning run.
