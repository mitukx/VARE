# Coding RL readiness and next end-to-end study plan

**Status:** proposal / engineering checklist. **As-of snapshot:** 2026-10-22 (JST).  
**Repository main at drafting:** `f29a8039e6bdce1aeb1849792b5f54a3c3d1fdd5` (PR #97 merged).  
**Do not treat this plan, or green CI alone, as a model-improvement result.**

## What already exists (and what it proves)

- The [continuous CPU control-loop benchmark](e2e-rl-continuation-v2-report.md) already covers **504 logical rounds, 513 actual optimizer attempts and 18/18 learner SIGKILL recovery comparisons**, for a small synthetic GPT-2 workload. Repeating 10–50 synthetic updates without a new hypothesis would not close the capability-evidence gap.
- [Asynchronous trajectory measurements](async-trajectory-runtime.md) establish a throughput/latency tradeoff on synthetic workloads. [GRPO CPU validation](grpo-e2e-cpu-report.md) checks numerical update arithmetic; it is **not** independent task improvement.
- The [actual-model update readiness audit](mlsys-actual-model-update-no-go-v1-report.md) found that the audited v1–v3 pool did not pass its preregistered uniqueness/prior-use gate. The subsequent [task candidate survey](../results/mlsys-task-candidate-survey-v1/) and deep qualification improve task provenance, but do not themselves produce an eligible held-out comparison.
- PRs [#94](https://github.com/mitukx/VARE/pull/94), [#95](https://github.com/mitukx/VARE/pull/95), [#96](https://github.com/mitukx/VARE/pull/96) and [#97](https://github.com/mitukx/VARE/pull/97) are merged. Their isolated-grader and calibration evidence validates narrow stated behaviors, not a full upstream test suite or policy learning.
- Existing Qwen/MLX-LM exploratory trials and one-example LoRA smoke are **feasibility observations**, not a controlled post-training benefit.

## Immediate integration and correctness queue

This table is a dated planning snapshot, **not live GitHub status**. Recheck current PR heads, reviews, reproducibility and CI before modifying or merging.

| Priority | PR | Required work and falsifiable exit criterion |
| --- | --- | --- |
| P0 | [#98: MLX-LM #1912](https://github.com/mitukx/VARE/pull/98) | Use the prospective v4 protocol; verify protocol hash and every grader implementation hash at each grading entrypoint; test varied argmax positions and nonzero retained values. **Exit:** on the pinned Apple Silicon/Colima environment, baseline FAIL, exact known-fix PASS, deliberately hardcoded index-0/zero outputs FAIL; all red-team probes and final-head CI pass. Store new append-only evidence. Keep historical v1–v3 reports intact. |
| P1 | [#99: MLX-LM #1971](https://github.com/mitukx/VARE/pull/99) | Vary `attention_value_scale` (not only 0.5); test the incremental **KV-cache** path with `make_cache()` so a wrong post-projection scaling implementation is rejected. **Exit:** baseline FAIL, known fix PASS, scale-ignoring and cache-incorrect mutants FAIL. Use a prospective protocol if locked files change. |
| P1 | [#100: Qwen feasibility](https://github.com/mitukx/VARE/pull/100) | Separate raw-format, valid-diff-but-unappliable, grader and infrastructure failures; reject trailing prose after a unified diff. **Exit:** adversarial parsing/classification tests and new prospective audit, with historic raw outputs and counts unchanged. |
| P1 | [#101–#108](https://github.com/mitukx/VARE/pulls) | Preserve stacked dependency order. Retarget one at a time only after the parent lands, audit implementation lock drift and check inherited P1/P2 findings. #108 remains Draft until its actual inference study gates are satisfied. |

**Non-negotiable:** a resolved review thread is not proof of an executed regression. Keep old calibration failures, timeouts, Docker exit 125 failures, unexecuted tests and previously exposed tasks classified as such. Never relabel historical evidence to make the metric look better.

## Proposed next study: actual model-coupled coding improvement

**Research question:** under a fixed, independently graded coding-task budget, does a locally feasible post-training policy outperform its no-update incumbent on a genuinely held-out set, *while* maintaining reliable asynchronous data/gradient provenance and crash recovery?

Two distinct estimands:

1. **Capability:** task-level held-out correctness on an independent grader, compared at matched generation/verification budgets.
2. **System:** useful valid trajectories and committed model updates per wall-clock second, queue-inclusive p50/p95 latency, verifier/stale rejection rates, exact optimizer/recovery state and duplicate-effective-commit rate.

### Gate A — data eligibility (hard NO-GO if missing)

- Inventory unique upstream tasks with immutable repository, baseline and fixed commit SHAs, test target hash, license, allowed edit scope and verifier.
- Exclude any task or paraphrased variants previously shown to either model or agent, including prior exploratory failures; log exposures **before** assigning train/dev/held-out.
- Freeze disjoint train/dev/held-out membership, prompt hashes and any public-test visibility, including explicit treatment of development tasks already exposed by previous studies.
- Apply the existing [readiness protocol](../protocols/mlsys_actual_model_update_no_go_v1.lock.json) and its minimum eligible-task gate. If it fails, **do not report a training comparison**; continue task qualification and document NO-GO.
- Held-out tests and oracles remain unavailable to the agent and to the optimizer. A generator-visible public test does not become a hidden independent outcome merely by running it in Docker.

### Gate B — independent grader certification (hard NO-GO if missing)

- Run pinned baseline FAIL / exact known fix PASS for each task; include adversarial wrong patches, shape/value shortcuts, invalid diff, stale-source, index-flag and Git replacement-object cases.
- Use isolated no-network, nonprivileged candidate execution with grader secrets, expected outputs and hidden oracles confined to the trusted parent.
- Verify frozen container digest, time/memory ceilings, manifest, protocol and code-file hashes **on each grading path**, not only the orchestration CLI.
- Calibration and red-team outcomes must be stored with a new versioned protocol and append-only records; do not modify earlier frozen results to accommodate new implementation hashes.

### Gate C — controlled model comparison (only after A and B)

- Pre-register the no-update, SFT and GRPO arms, any ablation, generation temperatures, seed IDs, generation budgets, stopping rules and exact task set. Freeze the evaluation plan **before running the arms**.
- Same starting checkpoint family and matched inference attempts, verifier calls and public-tool budgets; record training compute/cost separately.
- Select a small model/trainable adapter that fits the available Mac M1 Pro/32 GiB environment; measured feasibility precedes full runs. No paid API or GPU budget is assumed.
- Primary metric: held-out **per-task** probability of an independently correct patch at a fixed attempt budget. Treat tasks as statistical units; agent attempts within a task are not independent task observations.
- Report paired differences versus no-update with uncertainty intervals. Correct the two planned primary comparisons (SFT and GRPO) for multiplicity, e.g. Holm; also show raw task-level outcomes.
- Secondary: error classifications (syntax/format, patch-apply, independent grader fail, infrastructure, timeout), recovery reliability, freshness, artifact hashes, actual optimizer step counts and queue-inclusive latency. Do not optimize on these and then call them held-out capability gains.
- If the feasible held-out task count is too small to establish a meaningful bound on the planned effect, publish **feasibility only** with confidence intervals rather than success claims.

### Gate D — real end-to-end asynchronous execution

- Record for each trajectory: task/source/prompt identity, model policy version, verifier version, time of rollout/admission/use, trajectory state, status code, exact pre/post checkpoint identity and grader evidence.
- Compare controlled **synchronous** and **asynchronous** processing under equal compute/resource ceilings. Record staleness rejection and trajectories per accepted update; never report throughput without queue latency.
- Test at least three fault injection points: after rollout before admission, during candidate optimizer step, and after checkpoint/evaluation before promotion. Ensure restart never applies an already committed update twice, and report repeated *attempts* separately from effective commits.
- Bind promotion to independent checkpoint evaluation. On missing evaluator evidence, unverifiable restore, mismatched source hashes or inconsistent grader, **fail closed**.
- Existing 504-round synthetic control-loop evidence is the regression baseline. The *new* result must connect actual model weights, generated patches and independent held-out grading—not merely rerun a synthetic loop.

### Pre-registered stop/go table

| Condition | Action |
| --- | --- |
| Unique, unexposed eligible task pool fails readiness gate | **NO-GO for learning comparison**; improve data provenance only |
| Baseline/known-fix calibration or adversarial grader fails | **BLOCK grader**; no policy benefit claims |
| Local model or optimizer cannot complete bounded smoke within resource limits | **NO-GO for that model configuration**; select smaller legal alternative |
| Recovery mismatches optimizer/policy/checkpoint, or committed update duplicated | **BLOCK system claim**; retain and debug evidence |
| Eligible held-out comparison succeeds, with uncertainty reported | Report *task-scoped measured effect*, not broad frontier capability |
| CI is green but no live model-coupled outcome exists | Report engineering integration only |

## Artifact schema and lightweight preregistration

Draft a **new** run manifest before executing an experiment; do not reuse or replace previous run directories:

```json
{
  "schema_version": 1,
  "status": "PROPOSED_NOT_EXECUTED",
  "main_commit": "<40-char SHA>",
  "protocol_sha256": "<sha256>",
  "task_split_manifest_sha256": "<sha256>",
  "prior_exposure_manifest_sha256": "<sha256>",
  "model_checkpoint_sha256": "<sha256>",
  "adapter_sha256": null,
  "candidate_image_digest": "<sha256>",
  "arms": ["no_update", "sft", "grpo"],
  "seeds": [],
  "task_units": "independent_task_ids",
  "primary_metric": "heldout_correct_patch_per_task_at_fixed_attempt_budget",
  "matched_inference_and_verifier_budgets": true,
  "post_training_outcomes": null,
  "failure_classification": [
    "format", "patch_application", "grader_wrong_answer",
    "grader_infrastructure", "timeout", "not_attempted"
  ],
  "audit_status": "NOT_RUN"
}
```

All placeholder values are **illustrative**, not hashes or results. An actual run must fail if any required placeholder remains. Attach immutable input manifests, code/runtime provenance, raw attempt outputs, crash-injection events, reference optimizer/recovery checks, independent graders, raw per-task successes, confidence intervals and an audit log. Version new protocols rather than rewriting frozen reports.

## Short execution order

1. Finish #98 CI and live v4 calibration; publish append-only evidence, fix only defects supported by a regression, then conditionally merge.
2. Synchronize #99 against merged #98 and repair the two P1 false-positive pathways.
3. Synchronize #100 and repair failure taxonomies/raw-diff grammar; preserve historical outcomes as historical.
4. Reaudit the stacked PRs and avoid merging inheritance of unresolved P1/P2 findings.
5. Only after the data + grader gates are satisfied, preregister and run the **actual-model-coupled** E2E comparison.

**No training, model inference, benchmark run, or calibration was performed to produce this design document.**
