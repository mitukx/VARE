# Research mission journal v1

## Purpose and evidence boundary

`vare research` records a bounded, restartable mission ledger and returns a safe
next-action plan. It does not launch a model, grader, training process, worker
pool, scheduler, paid service, or background job. A later explicit invocation
must perform any planned work. This is a local journal, not a complete
autonomous research organization or a model-capability result.

The state machine is `DISCOVER → SCREEN → QUALIFY → FREEZE → EXECUTE → VERIFY
→ ANALYZE → DECIDE → INTEGRATE → NEXT_CYCLE → DISCOVER`. Each phase change
requires a pinned decision digest, completed action evidence, and no queued,
leased, or uncertain actions in the current phase. Task-family identity,
allocation, exposure, action identity, failure records, and attempt history are
append-only through journal events.

## Source and evidence integrity

Create a journal only from a clean Git checkout. Its source revision and local
runtime identities (Python, platform, SQLite) are checked on every mutation and
resume. The journal is initialized in a same-directory temporary SQLite file,
fsynced, then published with a no-overwrite hard link. SQLite transactions use
`synchronous=FULL`; events include a chained SHA-256 and a digest of the current
mission state. Event, task, action, payload, and database size limits fail
closed without pruning history.

Every completed action must reference a regular evidence file outside the
pinned source checkout, at most 512 MiB. `complete` recomputes its SHA-256 before
commit; `audit` and phase advancement recompute it again. Failure details are
retained inline up to 16 KiB, with an optional separately hash-checked file
reference. Replacing or deleting a referenced file makes audit fail. The
controller checks byte identity, not the scientific truth of the artifact.
When task provenance or a phase decision references an evidence file,
registration/transition requires its SHA-256 and later audits recheck those
same bytes.

The SQLite database and its hashes are not authenticated against a trusted
local owner who can rewrite the database and recompute the chain. The journal
assumes a trusted operator and local filesystem. Task certificates and
exposure-audit digests are recorded as operator-provided evidence; v1 does not
independently interpret or certify those documents. Resource reports are also
operator/executor inputs. Reservations bound concurrent claims and actual
reports are charged; the journal does not impose OS-level limits on an external
process. If a lease expires after work may have begun, it conservatively charges
the full reservation. An uncertain non-idempotent external operation is never
blindly retried.

Use one journal database for the entire multi-cycle research program: its
`NEXT_CYCLE → DISCOVER` transition retains prior task families and model
exposure. A separately initialized database does not share exposure history;
v1 therefore does not enforce cross-database task-family uniqueness. The
operator must retain and resume the same journal for a program, or supply an
independently verified exposure inventory when starting a distinct program.
Failure-streak escalation is bounded per action; repeated root causes across
different actions or future program cycles are not automatically clustered.

## Admission limits

- A model attempt is admitted only for one task registered as an eligible,
  unseen Development family and queued for `EXECUTE`. The exposure boundary
  durably consumes that task before inference. This records an operator
  admission; it does not prove the supplied certificate is valid.
- Training is disabled because v1 has no authenticated trajectory and reward
  readiness adapter.
- Held-out evaluation is disabled because v1 has no single-use cohort exposure
  and outcome adapter. The existing 15-family gate remains necessary and is
  not weakened.
- A model action that has crossed its exposure boundary is consumed even if
  its worker dies before reporting an outcome.

Failure classes are bounded: A retries only an explicitly retry-safe,
pre-exposure operation with backoff and at most three attempts; ambiguous
external state becomes `uncertain`; B supersedes an implementation; C consumes
the protocol/task outcome; D consumes a capability failure and points to a
fresh Development cohort; E blocks the gated experiment and allows independent
work to continue. Three repeated A failures with the same root cause require a
redesign. A retry uses the same action identity and a new fenced attempt; stale
lease generations cannot publish effective completion or failure.

## Use

```sh
vare research init --db /path/outside/checkout/mission.sqlite3 \
  --mission-id cycle-2026-10-26 --repo /path/to/clean/VARE
vare research task-add --db /path/outside/checkout/mission.sqlite3 \
  --mission-id cycle-2026-10-26 --record-json task-record.json
vare research action-add --db /path/outside/checkout/mission.sqlite3 \
  --mission-id cycle-2026-10-26 --record-json action-record.json
vare research resume --db /path/outside/checkout/mission.sqlite3 \
  --mission-id cycle-2026-10-26
vare research audit --db /path/outside/checkout/mission.sqlite3 \
  --mission-id cycle-2026-10-26
```

Completion requires a result-file reference, the file's SHA-256, and an
explicit JSON report for every resource key (`memory_bytes`, `disk_bytes`,
`model_tokens`, `wall_seconds`, and `candidate_attempts`). Model-attempt tokens,
memory, disk, and wall time are reserved before claim. Candidate attempts are
charged at the exposure boundary. The `resume` output states
`executor=not_configured`; it only returns an already queued action that is
safe under current phase, source, lease, and budget state.

## Verification and limits of the test evidence

Focused tests exercise the A–E policy, no-overwrite initialization, task/action
identity, development-only and phase gates, disabled training/evaluation,
resource reservation and charging, concurrent claims, stale fences, hash and
source corruption, retained artifact rechecking, reopen, and a `SIGKILL`
immediately after the committed model-exposure boundary. That kill test proves
the committed state is recovered and the task is not retried; it does not test
power loss, a kill during SQLite commit, external side-effect recovery, or a
background executor. The v1 result is local operational evidence only.

Before enabling training or held-out actions, a future version needs
machine-verifiable qualification records, trajectory/reward readiness checks,
single-use evaluation exposure, and an independently reviewed adapter. This
journal does not authorize those actions today.
