# Actual-model MLSys update readiness: NO-GO (v2 audit)

**Decision: NO-GO for a new no-update/SFT/GRPO held-out comparison.** The
current main inventory still has only 14 unique task definitions, and all
definitions in the frozen prior-use record are excluded from a new final
evaluation. The hard gate requires at least 15 untouched, independently
graded task definitions. The audit therefore stops before any new model
rollout or optimizer update. This is a task-readiness result; it does not show
that policy updates cannot improve task success.

## Repository and PR audit

The inventory audit is bound to VARE `main` commit
`312f499af5842faf5872ea554916e4d18e0b92cb`, tree
`80fa635de98791ac927a2e47cd2c591dd840f217`. The v2 lock pins this exact
commit/tree and the v2 auditor verifies that the task packs, v1 inventory
lock, and v1 auditor bytes in the working tree match that source commit before
running the inventory audit. The nested v1 audit also records its separate
embedded pack ancestry commit `46da14cd52777d99c0f2fd40c09d0513103d95e4`;
that commit is not represented as the VARE main revision. The v2 audit
reproduced `NO_GO`: 19 task entries, 14 unique definitions, maximum possible
unique inventory 14, 0 eligible held-out definitions, and 15 required.
Reproduce with:

```bash
python scripts/audit_mlsys_update_readiness_v2.py \
  --output results/mlsys-actual-model-update-readiness-v2/audit-v2.json
```

Relevant PR state was read from GitHub on 2026-10-20:

- #82, #84, and #89 are merged. #91 and #92 are merged readiness/report
  follow-ups. #90 is closed, and its task cohort is already consumed.
- #83 and #85–#87 remain open; #85–#87 are drafts, with several PRs stacked
  on prerequisite branches. The separated evaluator and manifest verification
  needed from that line are already present on current main through merged
  commits; do not rebuild those subsystems or treat unmerged PR metadata as
  additional task inventory.
- #88 is still open, conflicts with main, and has three unresolved review
  findings on head `ae6e469769127d042623f31dd8cc8247b1babba6`: a P1 nested
  shape-check omission in `global_grad_clip`, a P1 pack-manifest file
  verification omission, and a P2 worker task-resource-limit omission. The
  current main runner fixes the manifest inventory/hash issue by rebuilding
  the full file manifest in `verify_pack`; it does not import #88's candidate
  worker. The v1 global-gradient-clipping grader still has the reported
  nested-`zip` shape omission, so that task remains excluded from any new
  evidence. The pack is frozen and was not edited in place.

The original v1 inventory audit is also retained for comparison and is
reproducible with:

```bash
python scripts/audit_mlsys_update_readiness_v1.py \
  --protocol protocols/mlsys_actual_model_update_no_go_v1.lock.json \
  --output results/mlsys-actual-model-update-readiness-v2/v1-inventory-audit.json
```

The v1 audit checks task/evaluator hashes, pack commit ancestry, manifest and
split hashes, IDs, and prior-use classifications. Prior-use is a locked protocol assertion rather than a
reconstruction from raw rollout manifests; the independent 14-definition
inventory ceiling alone is sufficient for this NO-GO.

## Disclosed consumed-prompt replay

This audit inadvertently reran the frozen
`mlx_backend_microbenchmark_v1` prompt. A post-run review found that the
protocol and both backend output hashes are identical to the earlier retained
run at `results/mlx-backend-microbenchmark-v1/run-1`. That violates the
repository rule against rerunning consumed prompts. The repeated record is
retained and explicitly classified as a protocol violation; **it is not new
preflight evidence and is not used for the NO-GO decision**. The raw outputs,
summary, and inner hashes remain available for transparent accounting. The
duplicated observations were:

| Measurement | Metal GPU | MLX CPU |
| --- | ---: | ---: |
| Median decode throughput | 56.47 tokens/s | 56.72 tokens/s |
| MLX peak allocation | 2.41 GB | not reported |
| Process max RSS | 2.46 GiB | shared process measurement |

The original run, not this replay, is the valid source for the existing
feasibility claim in the v1 report. The benchmark does not separately time
prefill or sample utilization percentage. Raw repeated output, summary, pinned
protocol, and runner/evidence hashes are in
[`results/mlsys-actual-model-update-readiness-v2/metal-cpu-benchmark`](../results/mlsys-actual-model-update-readiness-v2/metal-cpu-benchmark/).

The earlier retained feasibility study remains the only actual update
measurement: Qwen2.5-0.5B MLX LoRA, 10 optimizer steps on one previously
verified development trajectory, finite loss from 1.281 to 0.807, 13.71 s
update wall time, and a reloadable 1.469 MB adapter. It is update-path
feasibility only. It has no matched no-update comparison, no SFT-vs-GRPO
comparison, and no fresh held-out task results. No new optimizer steps or
agent rollouts were run in this v2 audit because the frozen task gate failed.

## Validation

The source-bound v2 readiness audit passed. The combined readiness tests passed
5/5 (v1: 3, v2 commit/tree binding: 2), and Ruff passed for the new auditor
and its tests. A
local CI-equivalent full pytest attempt used Python 3.12 with external pytest
plugin auto-loading disabled. It was interrupted after 664 seconds at 34%
progress: 454 passed, 78 skipped, and 13 failures had been reported. The
Docker-backed failures were caused by Colima not sharing pytest's
`/private/var/folders/...` temporary directories into its Linux VM; a direct
bind-mount probe reproduced `bind source path does not exist`. A focused rerun
with `DOCKER_CONTEXT=colima` reproduced the same boundary (17 passed, 17
failed). The first unconfigured pytest launch also hit an unrelated host
Hydra/ANTLR plugin version conflict. These local attempts do not establish a
passing suite; Linux PR CI remains a required merge gate. The detailed local
attempt record is in `results/mlsys-actual-model-update-readiness-v2/local-validation.json`.

## Decision boundary and next gate

The duplicate prompt replay is not used to make a new hardware claim. This
audit does not repair the missing task cohort. Before another policy update, the next study
must produce and review a versioned pool of 30–60 fresh task candidates, freeze
15–20 untouched held-out tasks plus separate development and SFT data, and
pass a development-only model-success gate. Historical PR #84–#90 tasks,
including the defective global-gradient-clipping grader, cannot fill that
held-out set. Until those conditions pass, report no new capability result and
do not start SFT/GRPO training.

The original NO-GO and its limitation remain in
[`mlsys-actual-model-update-no-go-v1-report.md`](mlsys-actual-model-update-no-go-v1-report.md).
This report adds a source-bound inventory audit and discloses the consumed-
prompt replay; it does not replace or alter the frozen earlier protocol or
result.
