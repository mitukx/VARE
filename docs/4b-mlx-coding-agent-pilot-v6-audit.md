# 4B MLX coding-agent pilot v6: protocol deviation and evidence audit

**Audit date:** 2026-10-20
**Scope:** PR #89's frozen protocol `protocols/mlsys_coding_agent_4b_mlx_pilot_v6.json` and report `docs/4b-mlx-coding-agent-pilot-report.md`.

## Protocol deviation

The frozen protocol's `primary_metric` says the denominator is **10 task-seed attempts**. Its frozen task list contains five training tasks, while the inference arm fixes one seed (`101`) and `samples_per_task: 1`. The configured run therefore has five attempts, not ten. The report's 2/5 numerator and denominator are consistent with the configured five-task run, but the run is not protocol-faithful to the stated primary metric. This is recorded after the run; the protocol file remains unchanged and the deviation does not repair the preregistration.

The run is a five-task, one-seed development attempt only. The four held-out tasks remain outside this result. No held-out or generalization estimate follows from the pilot.

## Raw evidence availability

The report and original PR #89 description refer to an archive containing trajectories, candidate source snapshots, grader outputs, and a hash manifest. The archive and manifest are absent from the PR #91 tree and from its linked Apple Silicon evidence release. The protocol's configured output path is `$HOME/vare-evidence/4b-mlx-pilot/run-unique`; that path was not present on the audit host. A separate local VARE directory contained a 14B development run, which is unrelated and was not substituted.

As a result, this audit cannot verify the reported 2/5 pass count, per-task grader outcomes, token totals, timing, memory figures, or range-sums performance comparison. Those values remain historical author-reported observations only and are not admissible as verified research evidence in this PR. No archive was reconstructed from summaries or candidate descriptions.

## Reproduction status

The model, task pack, container, and command are pinned in the frozen v6 protocol. That protocol is not sufficient to reproduce this historical run because the exact run outputs are unavailable and the protocol denominator is inconsistent with its one-seed configuration. A future rerun must use a new protocol version, state the actual task-seed denominator before execution, retain all raw outputs, and publish a hash manifest. It must not reuse this pilot as held-out evidence.

## Claim boundary

This audit is a reproducible documentation correction, not a replacement experiment. It does not establish a positive or negative Qwen3-4B task-success result. The separate readiness NO-GO in `mlsys-actual-model-update-no-go-v1-report.md` remains based on its independently audited unique-task inventory ceiling and does not depend on PR #89's reported score.
