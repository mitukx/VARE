# Compute-constrained research roadmap

This roadmap is subordinate to the repository-wide operational contract in [`../AGENTS.md`](../AGENTS.md). Future automated agents should treat `AGENTS.md` as the first source of truth for invariants and claim policy.

VARE is optimized for a constraint that is common in independent research: abundant CPU/debugging time, limited accelerator budget, and a need for claims that survive external scrutiny. The roadmap therefore prioritizes artifacts whose value is not proportional to owned GPU-hours.

## North star

Build a closed improvement loop in which **real engineering failures become reproducible environments, trusted executable feedback becomes training/evaluation signal, and scarce learning compute is spent only where evidence says it is useful**.

The primary systems metric is not raw training scale. It is useful capability improvement per unit of scarce compute, with correctness, verifier freshness, and held-out promotion treated as hard constraints.

## Priority order

1. **Environment quality before learner scale.** Create repository-level tasks with immutable revisions, independent graders, replayable workspaces, and explicit failure taxonomies. A task is valuable only when success is externally checkable.
2. **Verifier security and provenance.** Evaluator commands are trusted configuration, candidate changes are diffed, evaluator assets are integrity-checked or mounted read-only, and task/result hashes are retained. Reward exploits are first-class failures.
3. **Trajectory evidence before RL claims.** Use fixed task packs to measure success, horizon, recovery, tool failures, verifier disagreement, and cost across agents. Publish negative/null outcomes.
4. **Failure-driven curriculum.** Convert recurring failure clusters into harder or more targeted environments. Prefer deterministic selection rules before learned task generation.
5. **Minimal learner experiments.** Use small public models or temporary/free accelerator access only to test a specific causal hypothesis. Pre-register the comparison, use multiple seeds when feasible, and stop if the experiment cannot distinguish the proposed mechanism.
6. **Upstreamable systems improvements.** When an environment exposes a scheduler, replay, inference, numerical, or observability bottleneck, extract the smallest generally useful fix and contribute it to the relevant open-source stack.
7. **Research automation last.** Only after the environment/evaluator loop is trustworthy should the system propose, implement, benchmark, and select its own engineering interventions.

## Evidence ladder under limited compute

- **E0 — harness correctness:** synthetic tasks; baseline fails, known fix passes, tampering fails closed.
- **E1 — real historical tasks:** immutable public repository revisions with independent regression tests; no model training required.
- **E2 — agent trajectories:** multiple agents/models on the same E1 tasks; CPU/API cost, success, recovery and failure evidence.
- **E3 — curriculum intervention:** failure-driven task selection beats a fixed/easy curriculum at equal rollout budget.
- **E4 — small-model learning:** a locked, multi-seed experiment shows held-out improvement or a clearly reported null/negative result.
- **E5 — systems impact:** measured throughput/latency/correctness improvement or an upstream contribution attributable to failures found by the environment stack.
- **E6 — recursive improvement:** the system repeatedly proposes interventions whose independent held-out effect remains positive after accounting for compute and verifier drift.

Claims must stop at the highest completed evidence level. Implemented code is not measured evidence.

## Budget discipline

Expensive experiments require a written decision rule before launch: expected information gain, maximum spend, primary metric, abort criterion, and what decision the result will change. If a CPU replay, historical-task study, or small-model ablation can falsify the hypothesis first, run that instead.

## Repository policy

The project documentation describes technical research goals only. External positioning belongs outside the repository; committed claims must be supportable from the artifacts themselves.
