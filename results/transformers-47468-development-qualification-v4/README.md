# Transformers #47468 Development qualification v4

V4 is a prospective correction to v3's execution boundary. V3 remains frozen at its original commit and source hashes, with no task outcomes. The independent exact-head audit of v3 found that its macOS profile permitted a forked process to call `setsid()`, leave the recorded process group, and continue writing inherited stdout/stderr after the runner reported group quiescence. V3 is therefore **NO-GO** for grader execution.

V4 reuses the immutable v3 journal, resource, source-preparation, and phase-quiescence code. A versioned wrapper adds a durable launch token and wraps interrupted-launch recovery; it does not edit the v3 file. The source lock pins both this v4 wrapper and the exact v3 core SHA-256. It uses the profile:

```text
(version 1) (allow default) (deny network*) (deny process-fork)
```

Before any task phase or freeze record, the runner executes a bounded probe under the exact profile. It requires a denied network connect, `os.fork()`, and Python subprocess spawn. A successful fork or spawn is a hard preflight failure. With process creation denied, a task subprocess cannot create the detached descendant that falsified v3's post-exit log hash barrier. The phase supervisor also retains v3's PID/create-time and process-group checks; a changed group remains fail-closed.

This profile is **not a container or general hostile-code filesystem sandbox**. `allow default` retains broad filesystem access. V4 does not claim containment against arbitrary filesystem writes, kernel defects, or every possible side effect. The v4 protocol is restricted to the trusted pinned source and public upstream grader qualification process; it does not authorize model-provided code execution. A separate exact-source independent audit must accept the v4 identity before any task grader phase.

The audit also found a separate P2 crash window: the parent can fail after launching a phase but before durably recording its PID/create-time. V4 now durably appends a UUID launch-token event before `Popen`, passes that token in the child's environment, and scans same-user processes during recovery. A unique match must also match the admitted test command and working directory; recovery then records PID/create-time/process-group identity and runs the pinned quiescence logic before hashing. A missing process is recorded as a scan result but remains deferred with no artifact hashes or inferred outcome. An inaccessible scan, multiple matches, command mismatch, or working-directory mismatch fails closed. This prospective recovery path requires independent adversarial review before any task phase.

The launch-token scan depends on the child retaining its environment marker. If a process clears that marker before controller failure, or same-user process metadata is inaccessible, recovery intentionally cannot prove cleanup and will not finalize hashes. The process-creation denial prevents descendants, but no general hostile-process or filesystem claim is made.

Before `--execute` or `--recover-interrupted`, V4 also compares the current source-lock SHA-256, protocol and runner hashes, Git commit and tree with the frozen manifest identity, and requires a clean worktree. This prevents an altered core plus edited lock from executing under an unchanged reviewed HEAD.

The identity gate reads and authenticates only the first `protocol_frozen` event before invoking any journal loader. If source identity fails while a pending event and torn manifest tail exist, both remain byte-for-byte untouched for later, authorized recovery.

Runtime state lives under `work/transformers-47468-qualification-v4/`: the append-only manifest is in `records/manifest.jsonl`, phase output is in `run/`, and recovered interruption records are in `records/`. The root `.gitignore` pins this exact runtime prefix so a fresh clone ignores the records; local `.git/info/exclude` rules are not accepted as provenance. `--execute` and `--recover-interrupted` reject output overrides outside the pinned `run/` directory. This keeps the reviewed source tree clean after freeze and during crash recovery.

The copied runtime inventory, dependency locks, task prompt, and exposure scan are byte-preserved from v3 and individually hashed. They are provenance inputs, not v4 outcome evidence. The runner, protocol, tests, and this README each receive new v4 hashes. V3 evidence and source are not edited.

## Checks allowed before independent review

These checks include only synthetic journal/runner tests, static checks, and the no-task preflight. They do not invoke the Transformers grader, baseline, mutant, fix, or a model.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_transformers_47468_qualification_v4.py
python -m py_compile scripts/run_transformers_47468_qualification_v4.py
python -m ruff check scripts/run_transformers_47468_qualification_v4.py tests/test_transformers_47468_qualification_v4.py
python scripts/run_transformers_47468_qualification_v4.py
```

The default command is a no-test preflight. Do not call `--freeze-record` or `--execute` unless the exact v4 commit, wrapper/core source lock, process-creation denial, and grader boundary pass a new independent audit. Approval or review of the v3 commit cannot transfer to v4.

## Scope of a possible result

The task is Development-only and its upstream PR, fix, and test are public. External model-training exposure is unknown. Even a complete baseline-fail, mutant-fail, fix-pass sequence would certify one source/grader task family only. It would establish neither model capability nor held-out validity, model-unseen status, task-family prevalence, or a learning benefit.
