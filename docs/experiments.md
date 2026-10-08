# Experimental program

VARE's experimental focus is post-training signal quality and policy updates under limited compute. A runnable learner is not evidence of learning; every result needs a frozen protocol, retained raw records, and an evaluation split that was not used for optimization.

## Current no-cost sequence

The first controlled synthetic run is retained in [`synthetic-dpo-report.md`](synthetic-dpo-report.md). Its finite-difference check and held-out NLL comparison are available, but the preregistered KL ceiling was exceeded and the reference-policy accuracy metric had a tie-handling defect. Treat it as a diagnostic non-pass. Continue with a separately versioned protocol; do not revise v1 after seeing its outcomes. Follow the staged plan in [`post-training-plan.md`](post-training-plan.md):

1. **Completed as a diagnostic non-pass:** v1 validated the objective gradient, but breached the KL ceiling and exposed a reference-accuracy tie bug.
2. Correct the tie metric and choose an update budget using training-only diagnostics; freeze a new protocol/seed cohort before evaluating held-out NLL, accuracy, KL, per-seed spread, and resource use.
3. Probe label noise and preference-distribution shift under a separate frozen protocol, preserving negative/null results and distinguishing optimization data from held-out evaluation.
4. Only if cached weights and local CPU capacity make it genuinely feasible, attempt a tiny adapter/DPO or RVL hook smoke test with strict time and memory abort limits. No model downloads, paid APIs, GPU rentals, or unbounded runs.

The synthetic stages test objective correctness and whether the harness can detect a known mechanism. They do not show language-model quality, reasoning, truthful behavior, broad preference alignment, or transfer to real users.

## Frozen L2 protocol

The first real-model campaign is frozen in `protocols/l2_rvl_qwen_v1.lock.json`. Do not edit or silently reinterpret it. It requires a 0.5B model, GSM8K, 3 arms × 3 seeds, 128 training and 256 held-out examples, 16 updates, and 8×8 rollouts. It remains **unrun** and is not feasible under the current no-GPU/no-spend constraint. Any future feasible model study must use a separately versioned protocol with an explicit budget and stopping rule.

Required fixed-compute arms in the frozen protocol:

1. uniform curriculum + fixed verifier;
2. failure-driven curriculum + fixed verifier;
3. failure-driven curriculum + joint policy/verifier freshness.

## Metrics and reporting

For a preference-learning run, report the frozen primary held-out metric plus preference accuracy, objective/loss, KL or another explicit policy-drift measure, reward/preference calibration where applicable, label-noise and shift conditions, per-seed outcomes, parameter delta, update count, wall-clock time, peak memory where measured, and exact software/data/protocol hashes. Do not treat training reward as capability. Do not report accelerator-hour efficiency when no accelerator was used.

## Evidence ladder

- **L0:** deterministic CPU demo implementation; the imported raw archive is corrupt and its historical summary is unverified. See `docs/evidence.md`.
- **L1:** objective/gradient checks and synthetic preference-policy updates; mechanism and harness evidence only.
- **L2:** a small real model on held-out tasks with multiple seeds; currently frozen as unrun and unaffordable under present constraints.
- **L3:** real rollout/trainer integration with measured throughput and lag; requires suitable free compute.
- **L4:** long-horizon environment with independent executable graders.
- **L5:** repeated intervention where reward is an independently measured downstream capability or systems improvement.

A feature is not evidence. Headline claims are allowed only when corresponding raw artifacts and the immutable protocol are retained.
