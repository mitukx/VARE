# Lag shift-score domain gate v1

Status: frozen before implementation.

Base commit: `dc57916272f32ebf89d63a7c45d9c16dd45d9fa6` (`origin/main`, 2026-10-15).

## Question

Does `LagController.assess()` fail closed when its total-variation shift-score input is malformed, while preserving admission and freshness for valid scores?

`DistributionShiftMonitor.score()` computes total variation and should return a finite value in `[0, 1]`. `LagController.assess()` is also a public boundary used to authorize replay freshness. A NaN currently evades `shift_score > max_shift`; its freshness calculation then returns `1.0`, so malformed evidence can be admitted. This gate tests that boundary. It does not claim that ordinary, unmodified engine execution produces NaN.

## Frozen cases

Use policy/verifier versions `5/5` and `9/9`, with `max_shift=0.2`, all other lag limits permissive, and no policy or verifier lag.

| Input shift score | Expected decision |
| --- | --- |
| NaN, positive infinity, negative infinity, `-0.01`, `1.01` | Reject; include `invalid_shift_score`; return finite freshness `0.0` |
| `0.0` | Admit; freshness `1.0` |
| `0.2` | Admit at the inclusive threshold; freshness `exp(-1.25 * 0.2)` |
| `0.21` | Reject; include `distribution_shift`; retain existing freshness formula |

## Acceptance

1. Save the baseline result and source hash before editing implementation.
2. Add direct regression tests for all frozen cases.
3. All new tests and the pre-existing lag/replay/engine-focused tests pass.
4. Valid-score decisions remain unchanged; invalid values are never admitted and never yield non-finite or positive replay freshness.
5. Run the repository's lint and type checks plus the full test suite before opening a PR.

This is a correctness-boundary repair only. It does not establish model-quality gains, a new RL algorithm, or an incident in the default distribution monitor.
