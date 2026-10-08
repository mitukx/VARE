# Verifier reward contract v1

## Question and hypothesis

Can a malformed structured `Verification` cross the verifier/learner boundary and become an optimization reward? The hypothesis was that mutable `Verification` objects returned by `FunctionalAttemptVerifier` or other `Verifier` implementations could bypass scalar score checks, be admitted to replay, and reach a learner without range or finiteness validation.

The frozen protocol is [`verifier_reward_contract_v1.json`](../protocols/verifier_reward_contract_v1.json), with its hash lock at [`verifier_reward_contract_v1.lock.json`](../protocols/verifier_reward_contract_v1.lock.json). It limits the claim to a deterministic CPU software contract; it does not measure training or policy capability.

## Baseline and intervention

At baseline commit `b74fe724be3161af2905a40ef4100a51ac6ca6c7`, the standard-library reproducer showed:

- A valid `Verification` mutated to `score=2.0` was passed to the training hook as reward `2.0`.
- A `NaN` score was passed to the training hook.
- `confidence=-0.1` did not prevent a valid-looking `score=0.75` from reaching the training hook.
- A valid control passed unchanged.

The reproducer and raw baseline are [`reproduce_verifier_reward_contract_v1.py`](../scripts/reproduce_verifier_reward_contract_v1.py) and [`baseline.json`](../results/verifier-reward-contract-v1/baseline.json).

The fix adds a shared validator/snapshot operation and enforces it on individual verifier outputs before aggregation, on the aggregate result, at the `CapabilityLoop` ingestion boundary before replay, and immediately before RVL constructs a trainer reward. It requires finite `[0,1]` score, confidence, and disagreement; exact boolean pass/trust flags; a nonnegative exact integer version; a nonempty name; and dictionary metadata. Weighted ensemble members must have finite positive weights. Invalid values are rejected; they are not clipped or silently repaired.

## Result

The unchanged reproducer now rejects the mutated `2.0` score, `NaN` score, and out-of-range confidence before the training hook. The valid control still delivers `0.75`. See [`postfix-local.json`](../results/verifier-reward-contract-v1/postfix-local.json). Regression tests also cover malformed fields, a custom verifier that bypasses the ensemble, and a reward mutated in replay before the RVL trainer; the latter must fail before `train_step`.

The local venv did not contain pytest, so the Python test suite could not be run locally. The dependency-free reproducer passed. Full regression status is pending CI and will be recorded in the follow-up evidence commit.

## Interpretation and limits

This is E0 harness/integration correctness evidence: one frozen exploit path was reproduced and closed with controls. The numeric range is an interface contract chosen by VARE; it does not prove that an in-range verifier reward is truthful, calibrated, hard to exploit, or correlated with task success. It does not establish improved model behavior, reward-model robustness, or the prevalence of malformed outputs in deployed systems. In-process code with arbitrary access to the trainer remains outside this validator's security boundary.
