# Actual-model MLSys update readiness: NO-GO (v2 audit)

**Decision: NO-GO for a new no-update/SFT/GRPO held-out comparison.** The
current main inventory still has only 14 unique task definitions, and all
definitions in the frozen prior-use record are excluded from a new final
evaluation. The hard gate requires at least 15 untouched, independently
graded task definitions. The audit therefore stops before any new model
rollout or optimizer update. This is a task-readiness result; it does not show
that policy updates cannot improve task success.

## Repository and PR audit

This audit was performed on current `main` at
`312f499af5842faf5872ea554916e4d18e0b92cb` after fetching `origin/main`.
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

The readiness audit is reproducible with:

```bash
python scripts/audit_mlsys_update_readiness_v1.py \
  --protocol protocols/mlsys_actual_model_update_no_go_v1.lock.json \
  --output results/mlsys-actual-model-update-readiness-v2/audit.json
```

It passed its integrity checks and returned `NO_GO`: 19 task entries, 14
unique definitions, maximum possible unique inventory 14, 0 eligible held-out
definitions, and 15 required. The frozen audit checks task/evaluator hashes,
pack commit ancestry, manifest and split hashes, IDs, and prior-use
classifications. Prior-use is a locked protocol assertion rather than a
reconstruction from raw rollout manifests; the independent 14-definition
inventory ceiling alone is sufficient for this NO-GO.

## Fresh local Apple GPU preflight

To avoid relying only on the previous feasibility report, the pinned
`mlx_backend_microbenchmark_v1` was rerun on this M1 Pro in a Python 3.12.12
environment with `mlx==0.32.2` and `mlx-lm==0.32.2`. It loaded the pinned
`mlx-community/Qwen3-4B-Instruct-2507-4bit` snapshot and confirmed Metal GPU
selection (`Device(gpu, 0)`, Metal available). Three greedy 96-token decode
repetitions produced identical output hashes on both backends:

| Measurement | Metal GPU | MLX CPU |
| --- | ---: | ---: |
| Median decode throughput | 56.47 tokens/s | 56.72 tokens/s |
| MLX peak allocation | 2.41 GB | not reported |
| Process max RSS | 2.46 GiB | shared process measurement |

The benchmark does not separately time prefill or sample utilization
percentage, so neither is claimed. Metal backend selection and allocation
provide evidence of GPU execution. Free swap was 806 MiB before and 695 MiB
after the run, remaining above the frozen 512 MiB stop threshold; free disk
remained about 203 GiB, above the 15 GB threshold. The small decode run did
not show a Metal throughput advantage. Raw output, summary, pinned protocol,
and the runner/evidence hashes are in
[`results/mlsys-actual-model-update-readiness-v2/metal-cpu-benchmark`](../results/mlsys-actual-model-update-readiness-v2/metal-cpu-benchmark/).

The earlier retained feasibility study remains the only actual update
measurement: Qwen2.5-0.5B MLX LoRA, 10 optimizer steps on one previously
verified development trajectory, finite loss from 1.281 to 0.807, 13.71 s
update wall time, and a reloadable 1.469 MB adapter. It is update-path
feasibility only. It has no matched no-update comparison, no SFT-vs-GRPO
comparison, and no fresh held-out task results. No new optimizer steps or
agent rollouts were run in this v2 audit because the frozen task gate failed.

## Validation

The readiness audit reran successfully, the evidence manifest hashes all
retained files, and `tests/test_mlsys_update_readiness_v1.py` passed 3/3. A
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

This rerun confirms that a pinned MLX/Metal inference environment is available
and the short decode workload fits the predeclared resource gate. It does not
repair the missing task cohort. Before another policy update, the next study
must produce and review a versioned pool of 30–60 fresh task candidates, freeze
15–20 untouched held-out tasks plus separate development and SFT data, and
pass a development-only model-success gate. Historical PR #84–#90 tasks,
including the defective global-gradient-clipping grader, cannot fill that
held-out set. Until those conditions pass, report no new capability result and
do not start SFT/GRPO training.

The original NO-GO and its limitation remain in
[`mlsys-actual-model-update-no-go-v1-report.md`](mlsys-actual-model-update-no-go-v1-report.md).
This report adds a fresh current-main audit and GPU preflight; it does not
replace or alter the frozen earlier protocol or result.
