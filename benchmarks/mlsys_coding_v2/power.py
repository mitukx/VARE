"""Paired held-out power for the draft v2 decision rule.

Task difficulty logits are ``b ~ N(mu, tau**2)``. The no-update arm draws
``k`` Bernoulli samples at ``p(b)``. Each of ``seeds`` training seeds draws
``k`` samples at ``p(b + delta_logit + N(0, sd_seed**2))``, and the trained
rate is the mean of those seed rates. The test is a one-sided paired t on the
``n`` per-task differences, alpha 0.05, with the critical values in ``TCRIT``.

The search reports the smallest logit step on a 0.1 grid whose Monte Carlo
power is at least 0.80, and the percentage-point lift of that step at the
simulated baseline. Type I is the same test at a zero logit effect and
``sd_seed`` 0. Running this file as a script consumes one ``Generator(0)``
in the order below. Importing it does not. The tabulated minimum detectable
effects are for one test. They are not Holm-adjusted.
"""

from __future__ import annotations

import numpy as np

# One-sided Student-t 0.05 critical values at df = n - 1 for n in {10, 15, 20, 30}.
TCRIT = {9: 1.833, 14: 1.761, 19: 1.729, 29: 1.699}
MU = -0.85
TAU = 1.5
SD_SEED = 0.3
ALPHA = 0.05
RNG_SEED = 0


def p_from(logits):
    return 1 / (1 + np.exp(-logits))


def power(
    rng,
    n,
    k,
    seeds,
    dl,
    mu=MU,
    tau=TAU,
    sd_seed=SD_SEED,
    reps=2000,
):
    hits = 0
    for _ in range(reps):
        difficulties = rng.normal(mu, tau, n)
        baseline = rng.binomial(k, p_from(difficulties)) / k
        trained = np.zeros(n)
        for _seed in range(seeds):
            effect = dl + rng.normal(0, sd_seed)
            trained += rng.binomial(k, p_from(difficulties + effect)) / k
        trained /= seeds
        differences = trained - baseline
        if differences.std(ddof=1) == 0:
            hits += differences.mean() > 0
            continue
        statistic = differences.mean() / (differences.std(ddof=1) / np.sqrt(n))
        hits += statistic > TCRIT[n - 1]
    return hits / reps


def percentage_points(rng, dl, mu=MU, tau=TAU):
    difficulties = rng.normal(mu, tau, 200_000)
    return 100 * (p_from(difficulties + dl).mean() - p_from(difficulties).mean())


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    baseline = 100 * p_from(rng.normal(MU, TAU, 200_000)).mean()
    print(f"baseline mean success {baseline:.1f}%")
    grid = np.arange(0.0, 2.01, 0.1)
    for n in (10, 15, 20, 30):
        for k in (8, 32):
            for seeds in (3,):
                mde = None
                for dl in grid:
                    if power(rng, n, k, seeds, dl, reps=1000) >= 0.8:
                        mde = dl
                        break
                type_i = power(rng, n, k, seeds, 0.0, sd_seed=0.0, reps=2000)
                mde_pp = float("nan") if mde is None else percentage_points(rng, mde)
                print(
                    f"heldout_tasks={n:2d} k={k:2d} seeds={seeds} "
                    f"MDE_logit={mde} MDE_pp={mde_pp:.1f} type1={type_i:.3f}"
                )


if __name__ == "__main__":
    main()
