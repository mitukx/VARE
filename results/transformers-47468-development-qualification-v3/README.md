# Transformers #47468 Development qualification v3

V3 prospectively supersedes v2 before any task outcome. V1 and v2 remain byte-for-byte unchanged and are both recorded as superseded before outcomes. V3 retains the same single Development-only public task and the same pinned source, grader, fix and mutant hashes. No model run or upstream grader outcome is included.

## Recovery and journal integrity

- Every append first writes and fsyncs a pending-event sidecar that pins the exact JSONL event bytes, their hash, sequence, prior event hash, and exact hash of the complete prior manifest prefix. The event is appended and fsynced before the pending sidecar is removed.
- Recovery repairs only a torn final suffix that matches the byte prefix of that durable pending event and whose complete prior prefix still verifies. It writes the exact suffix to a content-addressed quarantine file, then atomically replaces the journal with the unchanged verified prefix, the complete pending event, and a `journal_tail_repaired` record that includes the suffix path, SHA-256 and byte count. Corruption before the final line, a mismatched prior prefix, a mismatched pending record, or a suffix that does not match the intent fails closed without changing the manifest.
- Before recovery hashes phase artifacts, it verifies the recorded PID’s create-time and isolated process-group ID. It sends SIGTERM, waits up to two seconds, sends SIGKILL if live group members remain, then waits up to two more seconds. It fsyncs artifact files and hashes them only after the group has no live writers.
- If `phase_started` exists but no process identity was durably recorded, recovery appends a nonterminal deferred event and computes no artifact hashes. This handles the narrow launch window conservatively. The one-shot guard still prohibits another phase attempt.
- Recovery records partial outputs without treating them as grader outcomes. A recovery interruption can be retried for evidence repair, but the phase itself cannot be rerun.

## Runtime and task identity

V3 retains the v2 effective no-user-site inventory and rechecks the live interpreter executable/build plus exact host and overlay `pip freeze --all` bytes before any freeze-record or grader execution. The current pre-outcome check matched Python 3.12.12, 350 host distributions and 35 overlay distributions. The observed inventory files and hashes are under `environment/`; `records/runtime-inventory-observation-v3.json` records this check. No test was imported or collected for that check.

The task prompt remains public-task Development-only; upstream PR and test are public, external model-training exposure is unknown. This task cannot qualify as held out or model-unseen evidence.

## Deterministic checks

Run only the v3 unit suite, static check, and no-test preflight before independent review:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_transformers_47468_qualification_v3.py
python -m py_compile scripts/run_transformers_47468_qualification_v3.py
python scripts/run_transformers_47468_qualification_v3.py
```

The 15 unit tests use fake inventory subprocess results, temporary journals, and a detached fake Python writer that continues writing after simulated parent loss. They do not invoke the Transformers grader, baseline, mutant, exact fix, or a model. The default runner call is a no-test preflight and must report `PREFLIGHT_VALID_NO_TEST_EXECUTED`. Do not call `--freeze-record` or `--execute` until the exact committed v3 head and network/grader boundary pass independent review.
