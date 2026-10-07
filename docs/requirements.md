# Evaluation control-plane requirements

These requirements define technical acceptance under local CPU constraints. They do not declare deployment scale or model capability. Source and retained results are the evidence; each future measurement must freeze its own protocol first.

| ID | Requirement | Implementation | Regression or evidence |
| --- | --- | --- | --- |
| R1 | Scores bind to immutable task/grader and observed source/revision/diff | `runner.protocol`, `fingerprint`, `classify` | historical task calibration and provenance fault matrix |
| R2 | Bounded active jobs, deadline and combined captured output | `runner.execute`, semaphore | timeout, flood, child and inherited-pipe tests |
| R3 | Claims across local processes are serialized transactionally | `durable.Store.claim`, SQLite `BEGIN IMMEDIATE` | multiprocess claim-race test |
| R4 | Superseded or expired workers cannot publish current decisions | generation token + owner + live deadline predicate | expired/replaced lease and late completion tests |
| R5 | Retransmitted identical completion preserves one immutable attempt | stored payload equality, terminal attempt states | idempotence and conflicting completion tests |
| R6 | Process loss is recoverable within a finite attempt budget | lease expiry, token increment, exhaustion state | claimant death, rollback and retry-budget tests |
| R7 | Input freshness is recomputed before work, publication and export | Worker checks the claimed job before grading; completion checks that job; startup/export refresh all jobs | stale pending/reused result, changed-protocol tests and [frozen scaling evidence](freshness-scaling-report.md) |
| R8 | Invalidated history stays retained while current success is disabled | stale effective state, preserved attempt payload | duplicate-after-invalidation and direct-export tests |
| R9 | Public evidence replays the same terminal decisions | transactionally exported events/assets/attempts, strict audit | chain/summary/terminal transition tampering tests |
| R10 | Private runtime paths stay outside exported evidence | external live state, sanitized export | exported schema; live SQLite is not published |
| R11 | Agent/model findings are separated from synthetic mechanisms | evidence ladder and explicit reports | no agent/task-solving or model-learning claim from fixtures |
| R12 | No paid API, GPU or external compute is required | standard-library execution, public source fetch | experiment resource declarations and reproducible commands |
| R13 | Terminal completion does not repeat fingerprint work for unrelated jobs | `Store.complete` calls `refresh(job.id)`; full refresh remains at export | [cpu-refresh-scale-v2](../protocols/cpu_refresh_scale_v2.json), instrumented regression and offline bundle auditor |

## Acceptance rule

Implementation tests are necessary, but they do not establish an experimental result. The frozen recovery protocol adds three replications of eight declared fault cases and a historical-task integration check. Every current decision must replay from raw events, and every historical candidate outcome must match its pinned calibration. Failed and null experiments remain retained. Time to recover is observable but is not a performance headline under that protocol.

## Boundaries requiring future work

The state store coordinates processes on one host and one local filesystem. It does not provide network-partition tolerance, authenticated worker identity, multi-host consensus, machine power-loss validation or process containment after an uncatchable worker kill. Execution is at least once; a lease expiry can cause duplicate grading. Current terminal publication is fenced. Source files are checked before/after use and are not transactionally frozen with SQLite; trusted writers must keep them stable during grading. The [recovery contract](recovery.md) explains the guarantees and limits.
