# Experimental program

VARE's experimental focus is post-training signal quality and policy updates under limited compute. A runnable learner is not evidence of learning; every result needs a frozen protocol, retained raw records, and an evaluation split that was not used for optimization.

## Current no-cost sequence

The v1 and v2 synthetic runs are both retained. v1 remains a diagnostic non-pass: it exceeded its frozen KL ceiling and its old reference-accuracy field mishandled exact ties. v2 is a separate accepted confirmation with corrected tie scoring, a budget informed by a reported training-only sweep, a disjoint seed cohort, complete raw data, and a reconstructing audit. The exploratory sweep was not retained, and the formal development replay followed the confirmation; do not call v2 independently preregistered. The new noise/shift v1 protocol was committed before its run, and its audit reconstructs the held-out data and results. Do not revise any completed protocol after seeing outcomes. See [`synthetic-preference-robustness-v1-report.md`](synthetic-preference-robustness-v1-report.md) for its specific limits. Follow the staged plan in [`post-training-plan.md`](post-training-plan.md):

1. **Completed, synthetic only:** v1 exposed excessive policy drift; v2 used 100 updates, passed the finite-difference check, and met its recorded held-out NLL/KL acceptance rule on all 10 independent seeds. Its shuffled-ID arm is invalid and excluded from inference.
2. **Completed, synthetic only:** the noise/shift v1 study compared clean, 20%-flip and 40%-flip training at equal data/update budgets and evaluated against matched base/shifted preference labels. Its audit passed; flip arms had higher NLL on all seeds and the base-trained policy had higher mean NLL under the declared shift. This is one synthetic generator, not realistic reward robustness.
3. **Completed, non-pass:** the cached-model CPU study retained before/after outputs, policy drift and held-out evaluation. Its update failed the frozen rule; preserve it without tuning. Any future model run needs a distinct development cohort and fresh confirmation lock.
4. Seek an independent clean-clone reproduction and protocol review before adding further synthetic variants. Prefer a no-download task with an executable oracle only if it tests a materially new mechanism.

The synthetic stages test objective correctness and whether the harness can detect a known mechanism. The accepted v2 result is not evidence of language-model quality, reasoning, truthful behavior, broad preference alignment, or transfer to real users.

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
