#!/usr/bin/env python3
"""Post-hoc calibration-vs-raw diagnostic on the already-opened HH v2 cohort.

This analysis is descriptive only. It does not alter the frozen v2 decision,
authorize confirmation-row reuse, or create a v3 gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "results/cpu-hh-reward-model-v2/confirmation/run-1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def analyze(bundle: Path) -> dict:
    summary_path = bundle / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != "confirmation" or summary.get("decision", {}).get("pass") is not False:
        raise ValueError("expected the already-opened, non-passing v2 confirmation summary")
    cohorts = summary.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) != 3:
        raise ValueError("expected three retained v2 cohorts")
    n = len(cohorts[0].get("reward_margins", []))
    if n != 256:
        raise ValueError("expected the retained 256-prompt v2 confirmation cohort")

    paired_by_prompt = []
    raw_nll = []
    calibrated_nll = []
    for cohort in cohorts:
        raw = cohort.get("uncalibrated_reward_margins", [])
        calibrated = cohort.get("reward_margins", [])
        alpha = cohort.get("temperature_alpha")
        if len(raw) != n or len(calibrated) != n or not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError("margin arrays or temperature are invalid")
        if any(abs(float(c) - alpha * float(r)) > 2e-6 for r, c in zip(raw, calibrated)):
            raise ValueError("calibrated margins do not equal alpha times raw margins")
        raw_nll.append([softplus(-float(value)) for value in raw])
        calibrated_nll.append([softplus(-float(value)) for value in calibrated])

    for index in range(n):
        paired_by_prompt.append(float(np.mean([
            calibrated_nll[head][index] - raw_nll[head][index]
            for head in range(len(cohorts))
        ])))

    values = np.asarray(paired_by_prompt, dtype=np.float64)
    rng = np.random.default_rng(20261008)
    bootstrap_indices = rng.integers(0, n, size=(10000, n))
    bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975], method="linear").tolist()
    mean_delta = float(np.mean(values))
    sd = float(np.std(values, ddof=1))

    # Planning-only normal approximation. The SD is from this consumed cohort;
    # it is not a robust variance estimate for a new fixed-head protocol.
    planning = []
    z975 = 1.959963984540054
    for sample_n in (128, 153, 256):
        standard_error = sd / math.sqrt(sample_n)
        rows = []
        for assumed_delta in (-0.1942571341719062, -0.15, -0.10, -0.05):
            probability_upper_below_zero = 0.5 * math.erfc(
                -((-z975 * standard_error - assumed_delta) / standard_error) / math.sqrt(2.0)
            )
            rows.append({"assumed_true_delta": assumed_delta,
                         "normal_approx_probability_95pct_upper_below_zero": probability_upper_below_zero})
        planning.append({"sample_n": sample_n, "estimated_standard_error": standard_error,
                         "scenarios": rows})

    mde_80 = (z975 + 0.8416212335729143) * sd / math.sqrt(128)
    return {
        "analysis_status": "post_hoc_descriptive_only",
        "source_bundle": "results/cpu-hh-reward-model-v2/confirmation/run-1",
        "source_summary_sha256": sha256(summary_path),
        "source_protocol_sha256": summary.get("protocol_sha256"),
        "source_confirmation_already_opened": True,
        "prompt_count": n,
        "head_count": len(cohorts),
        "estimand": "mean across three fixed heads of per-prompt logistic NLL(calibrated margin) minus NLL(raw margin)",
        "mean_calibrated_minus_raw_nll": mean_delta,
        "paired_prompt_bootstrap_95pct": interval,
        "paired_prompt_sd": sd,
        "mean_raw_reward_nll": float(np.mean(raw_nll)),
        "mean_calibrated_reward_nll": float(np.mean(calibrated_nll)),
        "mean_length_baseline_nll": float(np.mean([c["baseline_metrics"]["logistic_nll"] for c in cohorts])),
        "per_head_mean_deltas": [float(np.mean(np.asarray(calibrated_nll[h]) - np.asarray(raw_nll[h])))
                                 for h in range(len(cohorts))],
        "planning_only_normal_approximation": {
            "sample_size_scenarios": planning,
            "approximate_effect_for_80pct_power_at_n128_two_sided_95pct_ci": mde_80,
            "caveat": "Uses the observed effect and variance from an already-opened v2 confirmation; optimism, cohort shifts, fixed-head recalibration, and training-cohort variation are not represented. Not a preregistered power analysis or a v3 gate."
        },
        "claim_boundary": "This secondary comparison does not change the frozen v2 non-pass, establish a calibrated reward model, or permit reuse of v2 confirmation rows. It only informs whether a fresh, separately frozen replication is worth considering."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.bundle)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
