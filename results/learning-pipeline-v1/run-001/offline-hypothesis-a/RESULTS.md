# Offline replay of hypothesis A: result

Verdict: **`not_supported`** under the decision rule in `PROTOCOL.md`. This is exploratory offline
evidence and a negative result. Following the plan, hypothesis A does not go to online RL on this
cohort.

Files: `PROTOCOL.md` (written before the confirm half was scored), `allocation_replay.py`,
`test_allocation_replay.py` (synthetic E0 checks: budget, cap, label blindness), `dev.json`, `confirm.json`.

```
python allocation_replay.py --rollouts ../rollouts/graded.jsonl --stage dev --out dev.json
python allocation_replay.py --rollouts ../rollouts/graded.jsonl --stage confirm --proxy short --variant exploit --out confirm.json
```

## Verified facts (from the logs)

* Cheap (`cheap_f2p_v1`) and trusted (`trusted_full_v1`) verdicts agree on all 160 train rollouts.
  In regime C, the cheap verifier alone reproduces every trusted label, so the trusted budget needed
  to build the run-001 RFT set is 0. Allocation has nothing to improve in that regime.
* Trusted grading took 108.0 s of a total of 1,690.6 s (generation 1,528.8 s, trusted 108.0 s, cheap
  53.8 s), i.e. 6.4%. Even a perfect allocation policy could save at most about 6% of rollout cost in
  this setup.
* No logprobs are logged, so confidence can only come from generation-side proxies.

## Dev selection (20 tasks; full-information RFT set 11 examples from 7 tasks; 47 gradable units)

The selection criterion was not written into the protocol. On dev, the `short` proxy and the
`exploit` variant score best both at the primary budget and averaged over the three budgets, so
either criterion gives the same choice. Even on dev, the selective policy did not beat the
plain confidence baseline that uses the same proxy. RFT recall:

| Policy | B=5 | B=9 (primary) | B=19 |
|---|---:|---:|---:|
| selective[short, exploit] | 0.364 | 0.455 | 0.545 |
| confidence[short] | 0.364 | 0.545 | 0.727 |
| confidence[agreement] | 0.219 | 0.364 | 0.833 |
| uniform | 0.144 | 0.261 | 0.541 |
| random | 0.136 | 0.239 | 0.494 |
| uncertainty | 0.063 | 0.107 | 0.246 |
| deterministic | 0.000 | 0.000 | 0.182 |

## Confirm (20 tasks; full-information RFT set 5 examples from 5 tasks; 29 gradable units)

| Policy | B=3 | B=6 (primary) | B=12 |
|---|---:|---:|---:|
| selective[short, exploit] | 0.200 | 0.200 | 0.600 |
| confidence[short] | 0.200 | 0.200 | 0.800 |
| uniform | 0.108 | 0.215 | 0.422 |
| random | 0.098 | 0.204 | 0.412 |
| deterministic | 0.000 | 0.200 | 0.400 |
| uncertainty | 0.148 | 0.200 | 0.370 |

Primary comparison: selective recall 0.200 against the best baseline (uniform) at 0.215, so
delta = -0.015. The task-bootstrap 95% CI is [-0.50, 0.00]. Because the upper bound is below 0.10,
the result is `not_supported`.

## Limits

* Small sample. The label-blind split put 11 of the 16 RFT examples in dev. Confirm has only 5
  examples, so recall moves in steps of 0.2.
* Conservative comparator. "Best baseline" is the maximum of five baselines, taken again in every
  bootstrap resample. This makes the CI conservative against the selective policy.
* Not blind. The aggregate run-001 summaries were known before the protocol was written. The
  within-log confirm half is an internal holdout, not an independent confirmatory dataset.
* Proxy outcome. RFT recall is a training-data proxy, not policy improvement. Label recovery does
  not show better learning, and none of these numbers show a policy effect.
* GRPO-group resolution (mixed groups fully labelled) is near zero at small budgets for every
  policy. It is recorded in the JSON files but is not informative here.

## Hypotheses (not verified)

* Hypothesis A could still matter where trusted grading is a large share of cost, or where cheap and
  trusted verdicts disagree. Neither condition holds in run-001. Testing it would need a different,
  independent distribution, without altering the original study.
* A proper confirmatory dataset would be fresh train rollouts on the 38 admitted train tasks that
  have no rollouts yet. Held-out tasks must not be used.
