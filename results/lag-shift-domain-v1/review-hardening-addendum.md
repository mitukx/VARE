# Lag shift-score overflow addendum

Status: frozen before the review-hardening patch.

Reviewed candidate: `cb58af1` (PR #53, based on `dc57916272f32ebf89d63a7c45d9c16dd45d9fa6`).

An independent review found that `math.isfinite()` raises `OverflowError` for a sufficiently large Python integer even though the value is outside the valid total-variation domain. Add one deterministic case, `shift_score = 1 << 4096`.

Expected behavior: return a denied decision with `invalid_shift_score` and finite freshness `0.0`; do not leak an exception. Existing frozen protocol cases and valid-score semantics remain unchanged.
