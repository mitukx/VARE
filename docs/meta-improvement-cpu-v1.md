# Improvement-value control for post-training: CPU v1

## Status and claim boundary

This is a **tabular expected-policy-gradient** experiment, not a language-model run, not an implementation of full recursive self-improvement, and not evidence of a novel algorithm. It demonstrates a narrow, executable **learn-from-independent-development-outcomes** loop for choosing post-training interventions with a fixed compute budget. No API, accelerator, or downloaded model is required.

The main research hypothesis for later, unimplemented work is whether a stateful improvement-value controller can choose between *actual* LLM policy updates, verifier audits, curriculum changes, and research-strategy modifications more efficiently than matched baselines without exploiting its evaluator.

## Mechanism

Three Bernoulli policies each select the correct action with probability `p=sigmoid(theta)`. A noisy verifier accepts a correct action with true-positive rate `tpr` and an incorrect action with false-positive rate `fpr`. Maximizing expected **proxy** reward has exact gradient

`d E[proxy] / d theta = p * (1 - p) * (tpr - fpr)`.

Each `train:skill` intervention executes `theta += 0.8 * gradient`. Initial tool verifiers have `tpr=0.69`, `fpr=0.85`; their apparent verifier reward can encourage wrong decisions. `repair-and-train:tool` spends three budget units to replace those rates with `0.94, 0.08` and executes one actual policy-gradient update. Other interventions have costs 1 or 2. The repair is a **fixed test fixture**, not a learned verifier update. All updates are exact expectations, without sampling variance.

An online ridge-regression UCB controller (`LinearImprovementModel`) sees only current policy probe probabilities, proxy acceptance rates, intervention cost, and prior **development** evaluation deltas. It predicts expected *development improvement per cost* for each intervention, selects a feasible intervention, and only then updates its estimate with bound-and-version-checked feedback. No confirmation score is supplied to selection or regression. Per-intervention models are contextual bandits, **not multi-step meta-RL** or a learned research agent. The controller's `Feedback` checks consistency but does not cryptographically attest grader independence; future integration must use protected evaluation evidence rather than caller-controlled numbers.

The objective is the independently computed clean success probability over three task families. Development weights are `(0.45, 0.35, 0.20)`; confirmation weights are `(0.35, 0.35, 0.30)`. Weights are distinct but defined by a shared synthetic family. Only initial and terminal confirmation are accessed for each run. A held-out seed is not a held-out real task distribution.

## Baselines and reproducible run

`round_robin` schedules through available intervention types; `proxy_greedy` maximizes current verifier acceptance divided by intervention cost and deliberately ignores independent correctness; `one_step_oracle` counterfactually chooses the best immediate **development** gain/cost (not the globally optimal long-term policy, and not a fair deployable baseline). All methods use the same initial parameters at each seed, and the same total budget of 32 units per episode.

From a Python 3.11+ editable checkout:

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest -q tests/test_meta_improvement.py
python3 scripts/run_meta_improvement_v1.py --seed-start 100 --seeds 40 --budget 32 --output /tmp/vare-meta-confirmation-v1.json
```

`--seed-start` changes the disjoint initial-condition seed window; do not retune on the confirmation window. The script writes all per-episode action traces, budget use, gain, and paired bootstrap differences. Each report is deterministic for the same runtime and source revision.

## Initial result (local CPU execution, 2026-10-13)

**Development-stage hyperparameter disclosure:** ridge and UCB exploration were explored on initial-condition seeds 0–19, including their synthetic confirmation metrics. The subsequently fixed configuration is `ridge=0.1, exploration=0.005`; the seed window 100–139 was not used in that exploration. Those holdout seeds still share the same hand-designed task/verifier simulator; this does not establish generalization to new problem families. The 95% intervals below are paired *seed bootstrap percentiles* (1000 draws), not uncertainty over environments or LLMs.

| Strategy | Mean terminal confirmation gain (40 seeds, fixed budget) |
| --- | ---: |
| Learned improvement value (contextual UCB) | +0.302322 |
| Round-robin | +0.274262 |
| Proxy acceptance greedy | -0.039193 |
| One-step development oracle | +0.263479 |

Learned value minus round-robin is `+0.028059` with paired-seed bootstrap 95% interval `[+0.027267,+0.028836]`. Learned value minus proxy-greedy is `+0.341514`, interval `[+0.339826,+0.343017]`. The development oracle underperforms here because it is one-step and optimizing different task weights. These numbers are synthetic and tuned to this simulator's action structure; the result should not be presented as a general advantage of UCB or RLVR.

## Important negative controls and next gates

- CPU unit tests check the analytic policy-gradient formula by finite differences, prove that the unrepaired false-positive verifier can **decrease** true success, check that repair can reverse its gradient, and enforce no duplicate/misbound feedback, model updates only after validated feedback, no confirmation reads during controller operation, budget limits, and deterministic paired reports.
- **Still absent:** model-generated solutions, real sampled GRPO trajectories, a learned verifier, counterfactual causal identification of improvement actions, cross-environment generalization, long-horizon meta-RL, and any independently confirmed LLM gain. The `selfimprove/` and `minigrpo/` modules are not changed by this experiment.
- **Next step:** adapt `Candidate` and `Feedback` to frozen `minigrpo` or `RVLGRPOHooks` study records, and place independent evaluation behind VARE's protected evaluator. Require a feasible model/task pairing, unchanged baseline, matched-update-count and token-budget controls, held-out tests never used for strategy selection, multiple seeds, and a genuine task-success gain before using RSI terminology for actual model behavior.

## Closely overlapping prior work

This prototype **does not claim novelty** over contextual bandits, online ridge regression, metareasoning, or dynamic curricula. Nearby 2026 research includes [Anticipating the Consequences of Curriculum Decisions](https://arxiv.org/abs/2610.04604), [MetaEvolve](https://arxiv.org/abs/2607.21971), [RAIL](https://arxiv.org/abs/2608.05080), [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677), and [VStress](https://arxiv.org/abs/2609.36958). A research contribution will require a new estimand or guarantee, concrete prior-art exclusion, and results on independently adjudicated real problems.
