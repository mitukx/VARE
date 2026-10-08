# Experimental program

VARE's experimental focus is post-training signal quality and policy updates under limited compute. A runnable learner is not evidence of learning; every result needs a frozen protocol, retained raw records, and an evaluation split that was not used for optimization.

## Current no-cost sequence

The v1 and v2 synthetic runs are both retained. v1 remains a diagnostic non-pass: it exceeded its frozen KL ceiling and its old reference-accuracy field mishandled exact ties. v2 is a separate accepted confirmation with corrected tie scoring, a budget informed by a reported training-only sweep, a disjoint seed cohort, complete raw data, and a reconstructing audit. The exploratory sweep was not retained, and the formal development replay followed the confirmation; do not call v2 independently preregistered. The new noise/shift v1 protocol was committed before its run, and its audit reconstructs the held-out data and results. Do not revise any completed protocol after seeing outcomes. See [`synthetic-preference-robustness-v1-report.md`](synthetic-preference-robustness-v1-report.md) for its specific limits. Follow the staged plan in [`post-training-plan.md`](post-training-plan.md):

1. **Completed, synthetic only:** v1 exposed excessive policy drift; v2 used 100 updates, passed the finite-difference check, and met its recorded held-out NLL/KL acceptance rule on all 10 independent seeds. Its shuffled-ID arm is invalid and excluded from inference.
2. **Completed, synthetic only:** the noise/shift v1 study compared clean, 20%-flip and 40%-flip training at equal data/update budgets and evaluated against matched base/shifted preference labels. Its audit passed; flip arms had higher NLL on all seeds and the base-trained policy had higher mean NLL under the declared shift. This is one synthetic generator, not realistic reward robustness.
3. **Completed, non-pass:** the first cached-model CPU update retained before/after outputs, policy drift and held-out evaluation. Preserve it unchanged.
4. **Completed, narrow model-level result:** a separate train-only development cohort selected 20 updates, then a frozen run on all 1,319 GSM8K test questions improved conditional preference NLL by 0.00697 nats/question over three fresh seeds. Accuracy stayed near chance. The adapter updates only two answer-label columns; see the [report](cpu-lm-gsm8k-dpo-confirmation-v1-report.md) and [audited bundle](../results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/).
5. **Completed development candidate:** a separate sequence-level GSM8K DPO update lowered verifier-preference validation NLL at four epochs while staying under the KL ceiling; generated exact-match was unchanged at 1/64. A decoder mismatch invalidated the v1 generation metric; v2 fixed it and passed a three-prompt Hugging Face parity audit. See the [v2 report](cpu-lm-gsm8k-sequence-dpo-development-v2-report.md) and [audited run](../results/cpu-lm-gsm8k-sequence-dpo-development-v2/run-2/).
6. **Incomplete confirmation v1:** the fresh 512-question cohort exceeded its 7,200-second CPU limit with two of three seeds complete. No aggregate result or pass decision exists; see the [incomplete record](cpu-lm-gsm8k-sequence-dpo-confirmation-v1-incomplete.md) and [partial audited bundle](../results/cpu-lm-gsm8k-sequence-dpo-confirmation-v1/run-1/).
7. **Confirmation v2 non-pass:** all three seeds lowered verifier-preference NLL with a paired 95% interval below zero and exact-match improved, but mean KL (0.7367) exceeded the 0.5 cap. The audited [report](cpu-lm-gsm8k-sequence-dpo-confirmation-v2-report.md) and [bundle](../results/cpu-lm-gsm8k-sequence-dpo-confirmation-v2/run-1/) retain all outcomes.
8. **Development v3 non-pass:** learning rate 0.0002 lowered KL to 0.2517 but mean free-form exact-match dropped below base (1.04% vs 1.56%); see the [report](cpu-lm-gsm8k-sequence-dpo-development-v3-report.md) and [audited bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v3/run-1/).
9. Change the completion/task setup using only fresh development rows. Do not tune against confirmation v2's held-out data. Freeze a further confirmation before reading its rows.

The synthetic stages test objective correctness and whether the harness can detect a known mechanism. The narrow forced-choice GSM8K result is evidence of a small conditional choice-probability shift. The sequence-level development is a verifier-preference validation signal only; its free-form accuracy did not move. No evidence here establishes general free-form language-model quality, reasoning, truthful behavior, broad preference alignment, or transfer to real users.

## Frozen L2 protocol

The larger real-model campaign is frozen in `protocols/l2_rvl_qwen_v1.lock.json`. Do not edit or silently reinterpret it. It requires a 0.5B model, GSM8K, 3 arms × 3 seeds, 128 training and 256 held-out examples, 16 updates, and 8×8 rollouts. It remains **unrun** and is not feasible under the current no-GPU/no-spend constraint. The separate cached-model forced-choice update is not a substitute for this trainer-integrated campaign or free-form capability evaluation.

Required fixed-compute arms in the frozen protocol:

1. uniform curriculum + fixed verifier;
2. failure-driven curriculum + fixed verifier;
3. failure-driven curriculum + joint policy/verifier freshness.

## Metrics and reporting

For a preference-learning run, report the frozen primary held-out metric plus preference accuracy, objective/loss, KL or another explicit policy-drift measure, reward/preference calibration where applicable, label-noise and shift conditions, per-seed outcomes, parameter delta, update count, wall-clock time, peak memory where measured, and exact software/data/protocol hashes. Do not treat training reward as capability. Do not report accelerator-hour efficiency when no accelerator was used.

## Evidence ladder

- **L0:** deterministic CPU demo implementation; the imported raw archive is corrupt and its historical summary is unverified. See `docs/evidence.md`.
- **L1:** objective/gradient checks and synthetic preference-policy updates; mechanism and harness evidence only.
- **L2:** one narrow small-model, forced-choice preference update is complete on CPU. Free-form generation, sequence-level preference training, and the larger frozen trainer-integrated campaign remain unmeasured.
- **L3:** real rollout/trainer integration with measured throughput and lag; requires suitable free compute.
- **L4:** long-horizon environment with independent executable graders.
- **L5:** repeated intervention where reward is an independently measured downstream capability or systems improvement.

A feature is not evidence. Headline claims are allowed only when corresponding raw artifacts and the immutable protocol are retained.
