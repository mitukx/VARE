# Evidence status

## L0 deterministic control-plane experiment

Command:

```bash
vare demo --rounds 8 --rollouts 512 --seed 7 --output evidence/l0_contextual_bandit_seed7.json
```

Observed in the committed seed-7 run:

- initial trusted held-out accuracy: **0.65625**;
- best promoted trusted held-out accuracy: **0.966796875**;
- promotions: **3 / 8** rounds;
- rejections: **5 / 8** rounds.

This result validates orchestration and fail-closed promotion behavior only. It is a synthetic contextual-bandit reference, not evidence of language-model self-improvement. L1+ claims require real-model experiments defined in `docs/experiments.md`.


## E0 executable engineering-environment harness

Command:

```bash
vare env-smoke --output-dir artifacts/env-smoke
```

Observed in v0.3.0:

- repository test suite: **25 / 25 passing**;
- intentionally broken numerical baseline: **fail**, score **0.05**;
- known harness oracle patch: **pass**, score **0.95**;
- protected evaluator-file mutation: **fail closed**;
- candidate workspace Git diff and task-spec SHA retained.

This is harness evidence only. The next required step is E1: immutable historical tasks from real repositories with independent evaluator assets.


## E0 fixed-budget campaign plumbing

A two-repeat campaign over the smoke task using a deterministic oracle editing command completed at **2/2 successes** with mean score **0.95**. Raw records are hash-linked through `records_sha256`. This validates repeatable workspace/agent/verifier orchestration only; the oracle is not a capability result.

## E1 historical task seed

`benchmarks/historical/rvl_behavior_policy_parity` pins pre-fix revision `e788f113ad6b246a361cd50a52eaf2867f2a66f8` and records the known fix revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72`. Its evaluator runs outside the candidate checkout. The local runtime used for this build cannot resolve external Git hosts, so the upstream revision was not cloned here; the task is therefore a **prepared E1 seed**, not a measured historical-agent result.
