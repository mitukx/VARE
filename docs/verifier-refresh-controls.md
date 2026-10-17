# Synthetic verifier-refresh controls

This bounded experiment tests the existing task-family `DistributionShiftMonitor` and existing refresh lifecycle contract. It adds no production refresh algorithm. The fixed [protocol](../benchmarks/verifier_refresh_controls_v2/protocol.json) precedes all outcomes; six runs use three seeds and two exogenous synthetic policy-shift scenarios. Each round has separate training observations, budgeted calibration labels and untouched independent candidate/baseline audit labels. All strategies see the same cohorts. Known synthetic truth and per-task IDs are retained in [compressed raw cohorts](../results/verifier-refresh-controls-v2/).

The verifier initially trusts a visible-test proxy. An exploit feature passes that proxy despite always being incorrect. Calibration fits a transparent feature/proxy truth table. Visible attacks change task family; hidden attacks remain within the original family. Periodic refresh occurs at rounds4/8; shift-triggered refresh uses existing TV>0.25. Both have a maximum128-label budget per run. Actual spend differs and is reported; no equal-cost efficiency claim follows.

| Scenario / strategy | Independent verifier error, mean ± sample SD | False proxy-only promotions (36 decisions) | Calibration labels per seed |
| --- | --- | --- | --- |
| visible / fixed | 17.86% ± 0.07% | 12 | [0, 0, 0] |
| visible / periodic | 0.00% ± 0.00% | 0 | [128, 128, 128] |
| visible / shift | 0.00% ± 0.00% | 0 | [128, 128, 64] |
| hidden / fixed | 18.36% ± 0.82% | 14 | [0, 0, 0] |
| hidden / periodic | 0.00% ± 0.00% | 0 | [128, 128, 128] |
| hidden / shift | 18.36% ± 0.82% | 14 | [0, 0, 0] |

All strategies missed zero independently measured improvements in this fixture. False promotions above are counterfactual proxy-only gate decisions relative to the independent truth gate, not actual checkpoint publication. Existing independent evaluation would expose them. The family-TV strategy succeeds on the visible shift and completely misses the hidden shift: a constant family marginal does not imply a reliable verifier. This is a controlled observability failure, not evidence that a new adaptive method improves model capability. No optimizer, LLM, asynchronous verifier deployment or real policy-verifier learning is measured.

Reproduce with Python3.11+ (new output directories):

```bash
python -m pytest tests/test_verifier_refresh_controls.py
python scripts/benchmark_verifier_refresh_controls.py --seed 79011 --scenario hidden --output /tmp/vare-refresh-hidden-new
```

Repeat visible/hidden for seeds79011/79012/79013. Raw rows, curves, summaries, frozen protocol, runner/monitor copies and per-file hashes are retained. Timing is real wall time of this small fixture on an interactive CPU host; it is not a throughput benchmark. The next useful observable is independently calibrated within-family verifier error under an actually qualified agent distribution, rather than another family-TV wrapper.

V1 data remain retained. Review exposed unpinned outcome-bearing imports during reproduction; v2 pins seven source dependencies before cohort generation and uses fresh seeds. The strategy, budget, threshold and acceptance rule are unchanged. A changed-mechanics negative control rejects before the experiment.
