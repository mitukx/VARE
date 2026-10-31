# Offline replay of hypothesis A: protocol (written before the confirm half was scored)

Status: exploratory, offline. The protocol was fixed after aggregate run-001 summaries were already
known (160 rollouts, 32 trusted passes, 12 tasks with a pass, 0 cheap false positives), but before
any allocation policy was scored on any task.

## Question

Hypothesis A asks: "At a fixed trusted-verification budget, can selective grading of training
trajectories lead to better downstream policy improvement than uniform or uncertainty-based
allocation?" This replay only tests the cheapest precondition: whether a selective allocation
policy recovers more of the training signal than strong baselines on the retained run-001 train
rollouts. A positive result here would not show better policy learning, because label recovery is
not policy improvement. A negative or inconclusive result means the mechanism should not go to
online RL on this cohort.

## Data

* `rollouts/graded.jsonl` of run-001 (160 rollouts = 40 train tasks x 4 samples, base policy; sha256 in
  `../SHA256SUMS.txt`). This file is hash-only and is read from a local path. No held-out data is used.
* Each rollout has a trusted label (`trusted_full_v1` all tests PASSED). Allocation policies cannot
  see it. A label is revealed only when the policy spends one unit of budget on that rollout.
* Split: a task goes to `confirm` iff `int(sha256("hypA-split-v1:" + instance_id), 16) % 2 == 1`,
  otherwise to `dev`. The split is label-blind and gives 20/20 tasks.

## Free, label-free structure (given to every policy)

* An edit that did not apply (`apply.applied` false) is a known failure and costs nothing.
* Rollouts with an identical `candidate_sha256` in the same task share one trusted run. Grading one
  reveals the others. This assumes the grader is deterministic.

## Regimes

* **N (trusted only):** policies see generation-side features only: apply status, finish reason,
  completion tokens, diff size, and within-task agreement of candidate hashes. No logprobs are logged.
* **C (cheap verifier on every rollout):** policies also see `cheap_f2p_v1`. This regime is reported
  descriptively, because cheap and trusted verdicts agree on all 160 rollouts. That makes trusted
  allocation trivial here, and nothing in this regime is interpreted as a mechanism result.

## Policies (regime N)

Each policy orders the gradable units, the distinct candidates of applied edits.

1. `random`: uniform random order.
2. `uniform`: round-robin over tasks, random within each task.
3. `deterministic`: log order (task-major, lowest `sample_index` first).
4. `confidence`: highest label-free confidence first. The confidence proxy is chosen on dev from
   {`agreement` = number of samples in the task sharing the candidate hash, `small_diff` = fewer
   changed diff lines, `short` = finish `stop` then fewer completion tokens}.
5. `uncertainty`: tasks whose candidates disagree most (more distinct candidates) first, so that
   mixed GRPO groups are resolved first.
6. `selective` (the hypothesis-A mechanism): adaptive and aware of the RFT admission cap. It uses
   revealed labels, never hidden ones. It grades the most confident candidate of every task first.
   After that it grades only tasks that still have fewer than 2 verified distinct passing outputs,
   ordered by variant: `exploit` (tasks with one verified pass first) or `explore` (tasks with no
   verified pass first). It never grades a task whose cap is full. The variant and the confidence
   proxy are chosen on dev.

Random tie-breaks are averaged over 400 seeds.

## Outcomes

* **Primary:** RFT yield, the number of examples the frozen admission rule admits from the verified
  passes. That is the sum over tasks of `min(2, distinct output_sha256 among verified passes)`.
  It is reported as recall of the full-information set at budget B = 20% of the gradable units of
  the scored half, rounded.
* **Secondary:** yield at B = 10% and 40%; tasks covered; mixed GRPO groups fully labelled.

## Decision rule (confirm half, primary budget)

Let `delta` = selective recall minus the best baseline's recall. The best baseline is the maximum
over the five baselines on the confirm half, which is conservative. The 95% CI comes from a task
bootstrap (2,000 resamples of confirm tasks).

* `supported_offline`: delta >= 0.10 and CI lower bound > 0.
* `not_supported`: CI upper bound < 0.10.
* `inconclusive`: otherwise.

`supported_offline` would only justify a small online test: train on budget-selected vs
baseline-selected RFT subsets with the same parameter budget. `not_supported` and `inconclusive`
mean no online RL for hypothesis A on this cohort.

## Context reported with the result

The trusted-grading share of rollout compute, from the logged seconds. A replication on fresh train
tasks is the proper confirmatory dataset: 38 admitted train tasks have no rollouts yet.
