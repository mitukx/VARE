# HH human-preference DPO v2: response-length diagnostic

**Status: post-hoc descriptive analysis.** This analysis uses the already-scored 256-prompt development cohort from [HH human-preference DPO development v2](hh-human-preference-dpo-development-v2-report.md). It changes no protocol, metric, gate, or frozen decision. The original DPO result remains a non-pass.

## Method

The analysis compares each preferred-versus-rejected **response-token count difference** (`chosen_tokens - rejected_tokens`) with the retained raw sequence-sum log-probability margin (`log P(chosen) - log P(rejected)`). It groups held-out pair accuracy by whether the human-preferred response was longer, equal in length, or shorter. DPO accuracy and margin are averaged across the three fixed seeds at each prompt before summarizing. The accompanying script verifies all 12 files in the source run manifest and checks the frozen protocol digest before calculating the summaries:

```bash
python scripts/analyze_hh_human_dpo_length_diagnostic_v2.py
```

The deterministic output is retained at [`length-diagnostic-v1.json`](../results/cpu-hh-human-preference-dpo-v2/analysis/length-diagnostic-v1.json). Its inputs are bound by the source summary, manifest, and protocol SHA-256 values recorded in that file.

## Descriptive results

| Response lengths in 256 pairs | Count | Base pair accuracy | DPO pair accuracy, mean across seeds |
| --- | ---: | ---: | ---: |
| Human-preferred response longer | 160 | 0.1063 | 0.1000 |
| Equal response-token count | 3 | 0.6667 | 0.5556 |
| Human-preferred response shorter | 93 | 0.9355 | 0.9462 |

A rule that always selects the longer response scores 0.6309 overall, exactly matching the run's reported length-only baseline. Across all 256 prompts, the Pearson correlation between the chosen-minus-rejected response-token count and raw sequence-sum margin was −0.956 for the frozen base and −0.956 for the DPO seed-mean margin. The DPO-minus-base margin change had correlation −0.275 with the same length difference.

## Interpretation and limits

On this development cohort, raw sequence-sum scoring strongly tracks response-length direction: it usually selects the human-preferred response when that response is shorter and usually misses when it is longer. DPO barely changes this pattern. The label set itself also favors the longer response often enough for the simple length rule to reach 63.1% accuracy. These results show why aggregate accuracy should be read alongside a simple length baseline and stratified diagnostics; they are consistent with substantial length sensitivity in the chosen scoring rule.

These correlations are descriptive and not causal. They were computed after the cohort had been used for the frozen development decision; there are no inferential intervals or independent validation here. They do not show that human preferences are generally length-driven, that length normalization would improve preference prediction, or that another policy objective would help. Do not replace the frozen v2 outcome with a re-scored result. Any follow-up must freeze length-aware metrics, strata, and advancement rules before opening a fresh cohort, and still needs an independent downstream task-success metric.
