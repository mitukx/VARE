# Documentation

Start with the [project summary](../README.md), then use the paths below to inspect architecture, implementation, experiments, and evidence. Historical reports and negative results remain available in the experiment and evidence indexes.

## New readers

| Question | Start here |
| --- | --- |
| What does VARE implement, and what has been demonstrated? | [README and selected results](../README.md#key-results) · [research scope and limitations](../README.md#research-scope-and-limitations) |
| How does the control plane fit together? | [Architecture and ownership](architecture.md) |
| How can I run a lightweight CPU check? | [Quick start](../README.md#quick-start) · [reproduction scripts](../scripts/README.md) |

## Systems engineers

| Topic | Implementation and evidence |
| --- | --- |
| Round ownership, evaluation cleanup, promotion, and recovery | [Engineering design](engineering-depth.md) · [whole-round recovery](runtime-round-ownership-v1.md) · [publication recovery](promotion-publication-recovery-v1.md) |
| Durable evaluation workers and leases | [Recovery contract](recovery.md) · [owner-loss report](grader-owner-loss-v1-report.md) |
| Async trajectory execution | [Runtime and measured tradeoffs](async-trajectory-runtime.md) · [trajectory capture boundary](trajectory-capture-boundary-v1-report.md) |
| Checkpoint and artifact limits | [Checkpoint read bounds](checkpoint-read-bounds-v1-report.md) · [artifact retention](artifact-retention.md) |

## RL researchers

| Topic | Evidence |
| --- | --- |
| Selected system and training-integrity results | [Benchmark guide](benchmarks.md) |
| Actual optimizer continuation and learner recovery | [Continuous CPU report](e2e-rl-continuation-v2-report.md) · [protocol](../benchmarks/e2e_rl_continuation_v2/protocol.json) |
| Group integrity and GRPO numerical reference | [GRPO report](grpo-e2e-cpu-report.md) · [protocol](../benchmarks/grpo_e2e_cpu_v5/protocol.json) |
| Independent evaluation and candidate promotion | [Architecture](architecture.md) · [RVL evaluation/promotion integration](rvl-evaluation-promotion-e2e-v1-report.md) |
| Research outcomes and candidate admission | [Experiment index](experiments.md) · [current gaps](current-gaps.md) |

## Agent researchers

| Topic | Evidence |
| --- | --- |
| Repository task execution, workspace binding, graders, and process ownership | [Agent evaluation guide](agent-evaluation.md) · [environment factory](environment_factory.md) · [MLSys coding pack v1](mlsys-coding-tasks-v1.md) |
| Workspace-bound evidence collector and its controls | [V2 report](verified_coding_agent_baseline_v2.md) · [v1 report](verified_coding_agent_baseline_v1.md) |
| Agent tasks and evaluation assets | [Historical benchmark tasks](../benchmarks/historical/) · [task preparation and script index](../scripts/README.md) |

Current agent evidence is limited to implementation and deterministic integration controls. No newly calibrated live-agent success cohort or model-capability result is established; the agent guide describes trust assumptions and unvalidated gates.

## Reproduction and auditing

| Need | Reference |
| --- | --- |
| Find a runnable reproduction or artifact auditor | [Script index](../scripts/README.md) · [selected benchmark commands](benchmarks.md) |
| Check protocols, raw outputs, and retained artifacts | [Protocol directory](../protocols/) · [evidence ledger](evidence.md) · [artifact retention](artifact-retention.md) |
| Review preserved negative, inconclusive, and superseded findings | [Experiment index](experiments.md) · [roadmap chronology](roadmap.md) |
| Find code-to-test-to-evidence links for design decisions | [Engineering deep dive](engineering-depth.md) |

## Historical reports

Detailed reports remain in `docs/` and are intentionally not all listed in the primary paths above. The [experiment index](experiments.md) and [evidence ledger](evidence.md) retain older studies, negative outcomes, corrections, and superseded records. Follow each report's own protocol, source revision, raw evidence, and limitations; PR and CI status statements inside old reports are historical snapshots.

**Evidence policy:** a deterministic regression validates a bounded implementation property; optimizer correctness does not show task improvement; and synthetic throughput is not production inference throughput. No retained real-model update demonstrates independently confirmed held-out task-success improvement.
