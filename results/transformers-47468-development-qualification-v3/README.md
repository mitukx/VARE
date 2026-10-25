# Transformers #47468 Development qualification v3

V3 prospectively supersedes v2 before any task outcome. V1 and v2 remain byte-for-byte unchanged and are both recorded as superseded before outcomes. V3 retains the same single Development-only public task and the same pinned source, grader, fix and mutant hashes. No model run or upstream grader outcome is included.

## Recovery and journal integrity

- Every append first writes and fsyncs a pending-event sidecar that pins the exact JSONL event bytes, their hash, sequence, prior event hash, and exact hash of the complete prior manifest prefix. The event is appended and fsynced before the pending sidecar is removed.
- Recovery repairs only a torn final suffix that matches the byte prefix of that durable pending event and whose complete prior prefix still verifies. It writes the exact suffix to a content-addressed quarantine file, then atomically replaces the journal with the unchanged verified prefix, the complete pending event, and a `journal_tail_repaired` record that includes the suffix path, SHA-256 and byte count. Corruption before the final line, a mismatched prior prefix, a mismatched pending record, or a suffix that does not match the intent fails closed without changing the manifest.
- Before any phase hashes artifacts, normal completion and recovery verify the recorded PID’s create-time and isolated process-group ID. They send SIGTERM, wait up to two seconds, send SIGKILL if live group members remain, then wait up to two more seconds. Artifact files are fsynced and hashed only after the group has no live writers. If normal completion finds and stops an orphan writer, that phase is aborted.
- If `phase_started` exists but no process identity was durably recorded, recovery appends a nonterminal deferred event and computes no artifact hashes. This handles the narrow launch window conservatively. The one-shot guard still prohibits another phase attempt.
- Recovery records partial outputs without treating them as grader outcomes. A recovery interruption can be retried for evidence repair, but the phase itself cannot be rerun.

## Runtime and task identity

V3 retains the v2 effective no-user-site inventory and rechecks the live interpreter executable/build plus exact host and overlay `pip freeze --all` bytes before any freeze-record or grader execution. The current pre-outcome check matched Python 3.12.12, 350 host distributions and 35 overlay distributions. The observed inventory files and hashes are under `environment/`; `records/runtime-inventory-observation-v3.json` records this check. No test was imported or collected for that check.

The task prompt remains public-task Development-only; upstream PR and test are public, external model-training exposure is unknown. This task cannot qualify as held out or model-unseen evidence.

## Rebuild the pinned public inputs

The source archive and grader/fix files live under the ignored `work/` directory so they are not mixed with the candidate or committed as trusted grader data. Rebuild them from the public immutable Transformers commits before running the preflight:

```bash
mkdir -p work/transformers-source-cache-v1 work/transformers-grader-v1/tests/utils work/transformers-47468-controls-v1
git init work/transformers-source-cache-v1
git -C work/transformers-source-cache-v1 fetch --depth=1 https://github.com/huggingface/transformers.git 6a994bc42f4c2c3e53ca384aff84750b85de0e6e
git -C work/transformers-source-cache-v1 fetch --depth=1 https://github.com/huggingface/transformers.git 3da1cdf80218c71ac4b4121cca43ea7def44e68a
git -C work/transformers-source-cache-v1 archive --format=tar 6a994bc42f4c2c3e53ca384aff84750b85de0e6e src/transformers > work/transformers-src-base.tar
git -C work/transformers-source-cache-v1 show 3da1cdf80218c71ac4b4121cca43ea7def44e68a:tests/utils/test_generic.py > work/transformers-grader-v1/tests/utils/test_generic.py
git -C work/transformers-source-cache-v1 show 3da1cdf80218c71ac4b4121cca43ea7def44e68a:src/transformers/utils/generic.py > work/transformers-47468-controls-v1/fix-generic.py
shasum -a 256 work/transformers-src-base.tar work/transformers-grader-v1/tests/utils/test_generic.py work/transformers-47468-controls-v1/fix-generic.py
```

The resulting SHA-256 values must match the locked source archive `2734b9db4f765e2fb1e69907600d4f992b463f108fa0eddcb2a3e679043791c0`, grader `a399b7719d561cad462c23d32460e1afa7e62535806fbd496a01adf8991dea70`, and exact fix `a97067244e842edc1fcaa77f0b57ea8e81d821105f6fe65311b77ccebf86f897`. The source archive is reproducible from the pinned upstream commit; the grader and fix remain outside the candidate tree.

## Deterministic checks

Run only the v3 unit suite, static check, and no-test preflight before independent review:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_transformers_47468_qualification_v3.py
python -m py_compile scripts/run_transformers_47468_qualification_v3.py
python -m ruff check scripts/run_transformers_47468_qualification_v3.py tests/test_transformers_47468_qualification_v3.py
python scripts/run_transformers_47468_qualification_v3.py
```

The 17 unit tests use fake inventory subprocess results, temporary journals, and detached fake Python writers that continue writing after parent loss, failed process-identity journaling, or normal leader completion. They do not invoke the Transformers grader, baseline, mutant, exact fix, or a model. The default runner call is a no-test preflight and must report `PREFLIGHT_VALID_NO_TEST_EXECUTED`. Do not call `--freeze-record` or `--execute` until the exact committed v3 head and network/grader boundary pass independent review.
