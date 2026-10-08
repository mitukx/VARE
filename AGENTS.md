# VARE agent charter

This file is the operational contract for AI agents and human contributors working in this repository. Read it before changing code, experiments, benchmarks, documentation, or claims.

## Mission

VARE exists to study **post-training signals and policy improvement under limited compute**. The system should make preference/reward signals auditable, test learning objectives and updates against independent measurements, detect when policy or verifier state has become stale, and spend scarce learning compute only when the expected information or capability gain justifies it.

The long-term target is not a large collection of features. It is a reliable loop:

```text
preference/reward/training signal
  -> policy update or targeted intervention
  -> independent held-out evaluation
  -> promote or rollback
  -> retained evidence
```

A successful VARE contribution makes some part of this loop **more correct, more measurable, more efficient, or more capable**.

## North-star question

> Under a fixed compute budget, which post-training signal or intervention improves an independently measured held-out outcome, and how do we know the update and evaluation are correct?

Prefer held-out preference accuracy or task success, reward/preference calibration, KL or policy drift, seed variance, label-noise sensitivity, and wall-clock/resource cost. Training reward alone is not a capability metric. Report accelerator-hours only when accelerator compute was actually used.

## Design stance

VARE is primarily the **outer improvement control plane**. Do not reimplement lower-level training/serving machinery merely to make this repository look self-contained when a tested external or sibling implementation already owns it. In particular, `Recursive-Verification-Lag` can provide token-exact rollout, GRPO, verifier execution, durable replay, verification-debt controls, weight synchronization, and GPU-oriented harnesses. VARE should integrate through explicit contracts and concentrate on failure selection, curriculum, freshness/shift decisions, evidence policy, and candidate promotion.

The repository must remain useful under compute constraints. CPU-only work is first-class when it creates durable evidence: historical tasks, independent graders, replay/provenance checks, failure mining, agent trajectories, curriculum experiments, correctness tests, profiling, and upstreamable systems fixes. Accelerator use is reserved for questions that cannot be answered by cheaper falsification first.

## Non-goals

Do not optimize for feature count, README impressiveness, benchmark cherry-picking, synthetic wins presented as model capability, or speculative claims about recursive self-improvement. Do not add wrappers whose only contribution is calling another model API. Do not create a second implementation of an existing subsystem unless the duplication is required by a clearly stated experiment or correctness boundary.

Do not shape repository documentation around external positioning goals. Public repository language should describe the technical problem, methodology, and evidence only; non-technical positioning belongs outside the codebase.

## Hard invariants

Treat these as correctness requirements, not preferences.

1. **Optimization and evaluation are separated.** Promotion evidence must be independent of optimization data whenever the protocol claims held-out improvement.
2. **Policy and verifier provenance are first-class.** Every trainable trajectory must have enough provenance to determine which policy and verifier produced or scored it.
3. **Freshness is recomputed at use time.** Data that was fresh when inserted can become stale after policy/verifier updates; cached freshness must never silently authorize training.
4. **Comparison groups stay intact.** Group-relative objectives such as GRPO must not silently lose or truncate members during replay selection.
5. **Candidate updates are transactional.** Snapshot incumbent -> train candidate -> evaluate incumbent and candidate under the same protocol -> promote or restore exactly.
6. **Executable evaluation fails closed.** Missing tests, evaluator corruption, protected-file changes, malformed output, timeout, or unverifiable provenance cannot count as success.
7. **Negative and null results are retained.** Never delete or hide a failed run merely because a later run succeeds.
8. **Protocols precede headline results.** For experiments used to support a research claim, freeze task selection, primary metric, seeds/budget, stopping rule, and acceptance rule before reading the outcome when feasible.
9. **Claims follow evidence, not implementation.** A runnable code path is not measured evidence. Stop claims at the highest completed evidence level.
10. **Raw artifacts are the source of truth.** Summaries and dashboards must be reproducible from retained raw records and hashes.

## Evidence ladder

Use this ladder when deciding what can be claimed and what to build next.

- **E0 — Harness correctness:** synthetic fixture; broken baseline fails, known fix passes, tampering fails closed.
- **E1 — Real historical task:** immutable public repository revision plus an independent regression/performance evaluator; no model training required.
- **E2 — Controlled post-training mechanism:** a preregistered CPU preference-policy experiment passes objective/gradient checks and evaluates a known synthetic preference distribution on disjoint held-out examples. This is mechanism evidence, not model capability.
- **E3 — Signal/intervention robustness:** matched-budget, multi-seed evidence tests preference noise, distribution shift, and a specified intervention against frozen controls.
- **E4 — Small-model learning:** a preregistered multi-seed experiment shows held-out model improvement, or an honestly retained null/negative result that falsifies a mechanism. Synthetic policy results do not count as E4.
- **E5 — Systems impact:** a measured correctness/throughput/latency/resource improvement or an upstream contribution caused by a bottleneck exposed by the environment/evidence loop.
- **E6 — Recursive improvement:** repeated system-proposed interventions continue to produce independent positive downstream effects after compute cost and verifier drift are accounted for.

Never relabel E0/E1 plumbing as E4 capability evidence.

## Priority order for future work

When several tasks are available, prefer the earliest unresolved item in this order unless evidence clearly says otherwise:

1. Preserve v1's failed outcome and v2's accepted synthetic confirmation; do not tune either protocol after observing outcomes.
2. Measure sensitivity to preference noise, distribution shift, reference-policy drift, and random seed under a separately frozen protocol; preserve null and negative outcomes.
3. Validate learner diagnostics and provenance (policy/reference/verifier versions, update count, KL, parameter delta, runtime, memory).
4. Audit whether a tiny cached-model adapter smoke test fits existing hardware and passes explicit time/memory abort limits; do not download model weights or spend money.
5. Use task trajectories, curriculum, and promotion infrastructure as supporting mechanisms for a clearly specified learning hypothesis.
6. Seek independent reproduction and broader grader coverage.
7. Increase model size or distributed scope only when a free resource is available and a cheaper experiment cannot answer the question.

## Compute discipline

Before starting an expensive run, write down:

- the hypothesis being tested;
- the cheaper alternative that was considered first;
- the primary metric and independent evaluation set;
- the maximum compute/API spend;
- the abort condition;
- the decision that will change based on the result.

If the result would not change the next action, do not spend the compute. If a CPU replay, historical regression, static analysis, synthetic counterexample, or smaller model can falsify the claim, run that first.

## Environment-task standard

A repository-level task should, where possible, contain:

- immutable source/revision provenance;
- a clear task statement without leaking the solution;
- isolated candidate workspace creation;
- evaluator assets outside or protected from the candidate workspace;
- correctness checks owned by the evaluator;
- performance/resource checks when relevant;
- explicit timeout/resource ceilings;
- task, evaluator, patch/diff, and result hashes;
- a known failing baseline and, for harness validation only, a known passing fix;
- a statement of what the task does **not** establish.

Prefer historical bugs, regressions, numerical failures, scheduler pathologies, recovery bugs, and verification failures that have an objective executable criterion.

## Agent-trajectory standard

When evaluating an external coding or research agent, keep the environment constant and record at least:

- task ID and immutable revision;
- agent/model identifier and configuration;
- random seed/repetition index when applicable;
- wall time and API/compute cost if observable;
- final diff and its hash;
- evaluator output and score;
- success/failure category;
- recovery attempts or tool failures when available;
- raw transcript pointer/hash when retention is permitted.

Do not change the evaluator or task after seeing one agent's result and then compare it against another agent under the changed task without versioning the task.

## Failure-driven curriculum rule

Curriculum logic should begin with transparent deterministic rules. Convert observed failures into typed clusters, then select tasks whose known attributes target those clusters. Learned task generation or learned curricula are later-stage additions and must be compared against simple baselines at equal rollout budget.

New generated tasks must pass verifier-integrity and non-leakage checks before entering a training or evaluation pool.

## Model-learning rule

Model updates are downstream of evidence, not the starting point. When a learner is introduced:

- preserve behavior-policy probabilities required by the objective;
- preserve group semantics;
- record policy/verifier versions;
- keep a trusted held-out metric outside the optimization reward;
- use transactional promotion/rollback;
- report negative, unstable, and constant-reward runs;
- compare against a simple baseline under matched compute.

A larger model is not automatically a better experiment. Scale only after the small experiment establishes that the mechanism is measurable.

## Upstream rule

If VARE exposes a generally useful bug or bottleneck in an external open-source system, prefer a minimal upstream fix over a VARE-only workaround when practical. A good upstream contribution has a reproducer, regression test, measured effect, and narrow scope.

## Documentation and claim policy

Write documentation as if a skeptical researcher will try to reproduce every sentence. Distinguish clearly among:

- implemented;
- unit-tested;
- synthetic/mechanistic evidence;
- measured on a real repository;
- measured on a real model;
- measured on real accelerator/distributed hardware;
- externally upstreamed or independently reproduced.

Avoid phrases such as "production-ready", "frontier-level", "self-improving", or "capability gain" unless the retained evidence directly supports the exact claim. Prefer precise descriptions of what was measured.

## Repository naming policy

Keep committed documentation, configuration names, workflows, benchmarks, issue templates, and code comments technically neutral. Do not add external-positioning language that is irrelevant to the research itself. Names of third-party open-source libraries or public models are appropriate when they are required to identify a dependency, baseline, or experiment.

## How an AI agent should choose the next task

Before making changes:

1. Read this file, `README.md`, `docs/roadmap.md`, and the relevant protocol/evidence files.
2. Identify the highest-value unresolved evidence gap, not the easiest feature to add.
3. Search for an existing implementation before creating a parallel subsystem.
4. State the falsifiable hypothesis or correctness property the change targets.
5. Add or update a regression test before or with the implementation whenever feasible.
6. Run the narrow test first, then the relevant suite.
7. Retain raw evidence and note limitations.
8. Update documentation only to the level supported by the new evidence.

If a requested change conflicts with a hard invariant, preserve the invariant and implement the closest valid interpretation.

## Definition of done

A change is not done because code was written. It is done when:

- the target failure or research question is explicit;
- tests or an experiment can falsify the change;
- relevant invariants remain enforced;
- evidence/provenance is retained when a measured claim is involved;
- the result, including null/negative outcomes, is documented without inflation;
- another agent can reproduce the next step from committed instructions.

The repository should become progressively **harder to fool, easier to reproduce, cheaper to learn from, and more capable on independent tasks**. That is the governing principle for future work.
