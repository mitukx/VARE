# Synthetic contextual preference optimization v2

## Question and scope

Does a small CPU policy trained on sampled pairwise preferences improve its preference predictions on independent contexts, while staying within a fixed policy-drift budget? The experiment uses a four-action linear-softmax policy and a synthetic Bradley–Terry teacher. It tests the objective implementation and the experiment harness. It says nothing about a language model, natural-language preferences, reasoning, truthfulness, alignment, or capability.

## Frozen procedure

The confirmation protocol is [`synthetic_dpo_cpu_v2.json`](../protocols/synthetic_dpo_cpu_v2.json), with its hash lock beside it. It uses 10 independent confirmation seeds, 512 training comparisons and 256 comparisons from separately generated held-out contexts per seed. The policy starts uniform. The clean arm receives the sampled training labels; the two diagnostic arms flip 20% of labels or shuffle labels. All arms use the same optimizer and update count. A central finite-difference check must pass before training.

The v2 protocol records that an earlier exploratory sweep on training metrics informed the 100-update budget. That sweep's raw bundle was not retained, so its timing and selection cannot be independently verified; do not describe v2 as independently preregistered. A separate 10-seed development bundle applies the stated rule—choose the largest tested update count with mean training-context KL at most 0.45 and every seed's training-context KL at most 0.50—and selects 100. That replay was produced after the confirmation run and is a reproducibility check, not the source of the confirmation budget. Its runner generates no held-out examples, and the development and confirmation seeds are disjoint.

The primary outcome is paired held-out preference NLL improvement over the no-update reference. Acceptance requires mean improvement of at least 0.05 nats per pair, a paired bootstrap 95% lower bound above zero, at least 8 of 10 seeds improved, and mean held-out KL no greater than 0.50. The rule was fixed in the protocol before the confirmation run.

## Results

The confirmation passed the versioned rule recorded with the run. All 10 seeds improved held-out NLL. Mean improvement was 0.307377 nats per pair; the paired bootstrap 95% interval was [0.292809, 0.322581]. Mean clean-policy KL to uniform was 0.387409, below the 0.50 ceiling. The finite-difference check passed.

| Arm | Held-out NLL | Pairwise accuracy | Brier score | Mean KL to uniform |
| --- | ---: | ---: | ---: | ---: |
| No-update reference | 0.693147 | 0.500000 | 0.250000 | 0.000000 |
| Clean preferences | 0.385770 | 0.828125 | 0.121186 | 0.387409 |
| 20% label flips | 0.492172 | 0.802734 | 0.159040 | 0.163665 |
| Invalid shuffled-ID arm | 0.720250 | 0.491797 | 0.262616 | 0.038622 |

The 20%-flip arm is a diagnostic observation from this generator and sample size, not a general robustness estimate. The v2 shuffle implementation shuffled absolute chosen action IDs across different action pairs; it could train on an action outside the example's pair. Its metrics are retained but invalid as a negative control and excluded from interpretation. The audit now detects this condition. The no-update accuracy is 0.5 because tied preference margins receive half credit.

## Reproduction and retained data

All development and confirmation examples, per-seed arm metrics, exact protocol snapshots, runner snapshots, environment details, and SHA-256 manifests are retained in [`results/synthetic-dpo-cpu-v2/`](../results/synthetic-dpo-cpu-v2/). The audit reconstructs all 40 seed-by-arm metric records from the raw examples, recomputes the paired bootstrap interval and acceptance decision, and verifies the manifests. The development audit reconstructs every candidate budget and confirms that no held-out examples were generated.

Reproduce into new output directories:

```bash
python3 scripts/develop_synthetic_dpo_budget.py --output /tmp/vare-dpo-v2-development
python3 scripts/audit_synthetic_dpo_development.py /tmp/vare-dpo-v2-development
python3 scripts/run_synthetic_dpo_v2.py --output /tmp/vare-dpo-v2-confirmation
python3 scripts/audit_synthetic_dpo_v2.py /tmp/vare-dpo-v2-confirmation
```

The scripts use Python's standard library. No model weights, accelerator, paid service, or network access were used for either run. The confirmation took about 6.4 seconds on the recorded local macOS arm64 system; local runtime is descriptive. The recorded Git revision predates the uncommitted v2 source and protocol files, so Git history does not independently timestamp the protocol lock relative to execution.

## Interpretation

This is a controlled synthetic contextual preference result under a versioned acceptance rule. With a uniform reference, the DPO-style loss reduces to pairwise logistic regression on four action scores; this checks a narrow objective/gradient and data-measurement path, not production token-level DPO. The task is small, the teacher and data generator are known, and the policy is not a language model. The next gap is a feasible, independently evaluated learner update with real model outputs; do not infer that this result transfers to RLHF, DPO on language data, reasoning, or deployed systems.
