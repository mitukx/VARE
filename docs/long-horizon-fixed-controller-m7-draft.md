# Long-horizon fixed-controller evaluation (M7) — DRAFT, UNFROZEN

> **Status: draft for review.** Nothing here is frozen or hash-locked, and
> nothing here supports a claim. The machine-readable draft is
> `protocols/long_horizon_fixed_controller_m7_draft.json`. No held-out task,
> hidden test, or grader configuration was opened while writing it.

## What exists now

`scripts/long_horizon_eval.py` runs multi-step episodes for one policy arm
under one fixed controller configuration:

* **Policy backend** — any object implementing
  `vare.learning_pipeline.policy.Policy`. `ScriptedPolicy` is a deterministic
  synthetic stub for CPU tests; `mlx_policy()` builds the existing `MLXPolicy`
  lazily (MLX is imported only when it is called).
* **Tools** — the `scripts/run_local_agent.py` tool surface and helpers
  (bounded listing/reading/search, `apply_patch`, `replace_text`, public tests,
  syntax check, finish gate). Test files are never in the readable scope.
* **Persistence and resume** — every turn is one `AgentStep` in
  `vare.trajectory_store.TrajectoryStore` (append-only, hash-chained). An
  interrupted episode is reclaimed after its lease expires; the workspace is
  rebuilt from the task factory, committed applied edits are replayed and
  hash-checked, and generation continues from the next unpersisted step. Work
  from an uncommitted turn is discarded, so no step is duplicated or lost.
  Enrollment records the controller digest, policy checkpoint, and task
  digest; resuming under a different one is refused.
* **Grading** — fail-closed. Only a result with schema
  `learning_pipeline_repo_env_grade_v2`, `executed: true`, and attested
  `required` isolation over `hmac_frames_v1` can be trusted. Missing, errored,
  untrusted, or unchanged-source episodes are recorded as not successful.
* **Metrics** — computed only from an audited store (`audit_trajectory_store`
  runs first; a tampered store is refused). Per episode: navigation, tool-call
  validity, edit application, public-test handling, recovery after failed
  actions, controller assists, tokens, tool invocations, timings, attempts,
  stop reason, and the primary outcome `final_success`. Aggregates include
  completion rate and success rate over all enrolled episodes.
* **Comparison** — `paired_comparison` pairs episodes by (task, seed), reports
  a repository-clustered paired bootstrap and a task-level exact McNemar test,
  and refuses policy comparisons across controller digests. A same-arm
  comparison across controllers is labelled `controller_ablation`.

Commands:

```
python scripts/long_horizon_eval.py smoke --output runs/lh --controller full
python scripts/long_horizon_eval.py smoke --output runs/lh --controller minimal
python scripts/long_horizon_eval.py compare runs/lh/stub-full runs/lh/stub-minimal
python scripts/long_horizon_eval.py run --arm base --policy-version 0 \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit --controller full \
    --manifest configs/learning_pipeline_v1/train.manifest.jsonl \
    --cache <cache> --limit 2 --output runs/lh-train \
    --grader-factory <module:function>
```

The smoke fixture and stub are synthetic and carry `synthetic` flags in every
episode. Without `--grader-factory` the `run` command records every episode
as `grader_unavailable` (never a success).

## Draft design (to be settled before freezing)

* **Arms** — `base` (required), `rft` (only if the v1 adapter exists with its
  own evidence), `grpo` (only if produced under its own frozen protocol),
  verifier-aware arm only with a separate written justification.
* **Controller** — `full` preset is primary; `minimal` keeps budgets and turns
  off normalization, reminders, bare-edit recovery, automatic public tests,
  and the finish gate.
* **Tasks** — smoke on the Train manifest only. The held-out set is a
  repository-disjoint set that must be selected and hash-frozen before any
  access; it is deliberately left undefined here.
* **Primary outcome** — trusted-grader pass per (task, seed), all enrolled
  episodes in the denominator.
* **Statistics** — paired by (task, seed), repository-clustered bootstrap
  for the difference; exact McNemar on task-level outcomes as support.
* **Ablations** — policy (controller fixed) and controller (policy fixed),
  reported separately.

## Will not be claimed

* Anything from the synthetic fixture or scripted stub.
* Capability changes from Train-split runs.
* Policy effects from comparisons across controller digests.
* Success from public tests, self-report, or ungraded episodes.

## Not yet verified

* Real MLX generation through the harness (needs the target Mac).
* The `repo_workspace_factory` path on real pinned clones and the
  `RepoEnvTrustedGrader` adapter against the isolated grader (that grader is
  not on `main` yet).
* Per-episode wall-clock on the target machine, which sets task counts.
