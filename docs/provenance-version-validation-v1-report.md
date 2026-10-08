# Provenance version validation v1

## Finding

**Fixed and covered by CPU regressions.** Before the change, VARE clamped negative policy/verifier lag to zero. A future rollout policy version or verifier version was therefore represented as fresh. The RVL SQLite reader repeated the same behavior in both its snapshot and ready-experience paths.

At commit `d7ec900`, a ready RVL group with recorded policy/verifier version 6 and current version 5 produced `admitted=true`, no lag reasons, snapshot lags `0/0`, and one ready experience with lags `0/0`. The exact reproduction and pre-fix source hashes are retained in [`baseline.json`](../results/provenance-version-validation-v1/baseline.json).

## Contract after the fix

- Versions must be exact non-negative Python integers. Booleans, floats, strings, and negative values fail closed.
- `LagController.assess` rejects a future policy or verifier version with `policy_version_ahead` or `verifier_version_ahead`. Invalid values get `invalid_policy_version` or `invalid_verifier_version`.
- RVL snapshots and ready-experience loading raise `ValueError` for future stored policy/group-verifier versions. Ready payloads with per-experience verifier overrides are also checked.
- A pending-verification group may retain its existing `-1` verifier sentinel in snapshots; it is not treated as a scored ready item. Existing ready data with an ordinary stale version still reports its nonnegative lag.

## Verification

`tests/test_lag.py` covers policy/verifier future versions, negative values, and malformed types. `tests/test_rvl_integration.py` covers future group policy/verifier versions, future per-experience verifier overrides, malformed boolean overrides, and preserves the existing ready/pending fixture. The prior baseline reproducer now raises instead of returning a fresh item.

This is a narrow data-integrity correction, not an empirical estimate of provenance error rates, a change to the learning objective, or evidence of model improvement. CI outcome will be appended after the pushed revision completes its full repository suite.
