# Evaluation statistics

`src/vare/eval_stats.py` provides four calculations used by a report. None of them accepts a model update.

- Unbiased pass@k for one task is `1 - C(n - c, k) / C(n, k)` when at least k failures remain, and 1 otherwise. The task-level figure is the mean of those estimates.
- The paired bootstrap resamples tasks with replacement and returns a percentile interval for the mean of `right - left`.
- McNemar's exact p-value is twice the lower binomial tail of the discordant counts, clamped to 1.
- `mcnemar_power` is a normal approximation. For `n = 100`, discordant probabilities `0.2` and `0.05`, and alpha `0.05`, it returns `0.8621993605389122`. A zero difference returns alpha.

The empty report form is `docs/templates/paired-comparison-report.md`.
