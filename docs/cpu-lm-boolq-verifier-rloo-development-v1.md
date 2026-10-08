# BoolQ binary verifier-RLOO development v1

## Status

The frozen CPU-only development run completed on five seeds and the corrected same-host auditor passed. Its frozen advancement gate is a **non-pass**. The first automatic audit attempt exposed an auditor bootstrap bug; the original failure record is retained, the bootstrap was regression-tested and fixed, and the same stored run was independently replayed without retraining. Even a development pass would be only a screening result; confirmation requires a new lock and a separately audited execution.

The cohort selector hashes `question` and `passage` fields across each official split, including the reserved validation rank range. Earlier BoolQ v16/v17 code also hashed validation prompts. Therefore, the reserved rows below are unscored and unused by this follow-up; they must not be described as never text-hashed or never opened.

## Question

On a frozen small language model and a passage-grounded Yes/No task, can a binary verifier-reward contextual-bandit update improve development balanced accuracy under a fixed CPU budget? How does sampled RLOO behave against exact expected reward and supervised controls when the expected reward gradient is identical?

This is not sequence-level RL. The policy selects one of two next-token actions after a frozen prompt. The transformer and language-model head remain frozen; a zero-initialized affine delta modifies only the Yes-minus-No action margin.

## Locked design

- **Source data:** `google/boolq`, pinned revision and Arrow hashes in [`protocols/cpu_lm_boolq_verifier_rloo_development_v1.json`](../protocols/cpu_lm_boolq_verifier_rloo_development_v1.json). Rows are selected by SHA-256 rank over question and passage text.
- **Model:** cached `Qwen/Qwen2.5-0.5B-Instruct`, pinned revision. The protocol hashes the model weights and tokenizer/configuration files; execution is offline and refuses missing or changed cache files.
- **Training cohort:** train ranks `[280, 408)`, 128 rows. Training labels enter only through the deterministic answer-key verifier reward.
- **Development cohort:** validation ranks `[1536, 2048)`, 512 rows. Labels are used only for checkpoint selection and screening metrics.
- **Reserved interval:** validation ranks `[2048, 2560)`, 512 prompt hashes. This run does not request labels, construct prompts, tokenize, score, or serialize these rows.
- **Matched arms:** frozen base, scalar calibration, contextual supervised action training, exact expected verifier reward, and RLOO with four fresh Bernoulli actions per prompt/update.
- **Optimizer:** five fixed seeds, identical pre-generated minibatches across trained arms, 100 updates, a four-thread CPU ceiling, a six-GiB peak-RSS ceiling, and a one-hour wall-time ceiling for the study runner. The separate audit process has its own one-hour and six-GiB ceilings; these are per-process limits, not a combined end-to-end budget. No GPU, paid API, or network access is allowed.
- **Primary screen:** selected RLOO minus frozen-base development balanced-accuracy gain of at least five percentage points, paired prompt-bootstrap lower bound above zero, at least four of five seeds no worse than base, a viable base floor, and the locked KL ceilings. Checkpoints are selected on development, so the interval is descriptive and cannot serve as confirmation.

## Objective and controls

For deterministic correctness reward `r(a)=1[a=y]` and class weight `c_y`, the expected RLOO score-function gradient is `c_y * grad(p_pi(y|x))`, the same gradient as exact expected verifier reward. RLOO is not assumed to provide a new reward signal or algorithmic superiority. The exact expected-reward arm measures finite-sample optimization behavior; contextual SFT measures whether sampling the verifier reward adds value beyond action-label supervision. A scalar-only calibration arm checks whether class-prior adjustment accounts for any apparent gain.

The runner performs an exact-enumeration gradient self-check across both labels, several action probabilities, and reward weights before it reads the selected cohorts. The advancement gate requires that check to pass.

## Independent audit

The runner launches a separate offline auditor before accepting a result bundle. It checks source snapshots against the clean pre-run commit, verifies the pinned local dataset/model/tokenizer files, reconstructs hash-ranked cohorts and labels, checks prompt separation from earlier BoolQ runs, and uses prompt columns only for the reserved interval. It then reruns tokenizer/model feature extraction for selected train and development prompts, recomputes training-only normalization, independently reconstructs metrics and bootstrap intervals, reproduces minibatch and RLOO action RNG streams, verifies rewards/LOO advantages/log probabilities, and replays every adapter update through each checkpoint.

The audit is same-host reproducibility using the pinned local model and CPU software stack. It is not external reproduction or a second implementation of the transformer. The accepted bundle's final SHA-256 manifest is written after the audit report and includes `audit.json`; the auditor first verifies the pre-audit manifest. Study and audit processes each obey their own frozen CPU, offline, wall-time, and memory limits.

## Claim boundary

Regardless of outcome, this experiment concerns one public reading-comprehension dataset, one cached 0.5B model, a two-action verifier-reward contextual bandit, and a small CPU budget. BoolQ may have appeared in model pretraining. It cannot establish free-form generation improvement, sequence-level RL, human-preference alignment, broad reasoning, general capability, or scale. A non-pass is retained without changing the protocol; a pass requires a new confirmation protocol before any reserved answer label is accessed.

See the machine-readable [protocol](../protocols/cpu_lm_boolq_verifier_rloo_development_v1.json), [lock](../protocols/cpu_lm_boolq_verifier_rloo_development_v1.lock.json), [runner](../scripts/run_cpu_lm_boolq_verifier_rloo_development_v1.py), [auditor](../scripts/audit_cpu_lm_boolq_verifier_rloo_development_v1.py), and [tests](../tests/test_boolq_rloo.py).

Measured results and retained artifacts are in the [development report](cpu-lm-boolq-verifier-rloo-development-v1-report.md).
