# HH-RLHF v2 post-hoc calibration diagnostic

## Status

This is a secondary analysis of the already-opened v2 confirmation bundle. It is not a new experiment, does not alter the frozen v2 decision, and cannot support a confirmatory claim. The original v2 joint gate remains a non-pass because its confirmation reward-head-versus-length NLL interval crossed zero.

## Question and computation

For each of the three unchanged v2 reward heads and each of the 256 retained confirmation prompt contexts, compare logistic NLL for the raw pairwise reward margin with NLL for the already temperature-scaled margin. First average the calibrated-minus-raw NLL difference across the three heads within each prompt, then bootstrap prompt contexts (10,000 percentile resamples, seed `20261008`). The diagnostic reads only `summary.json` from the retained confirmation result; it does not parse any new dataset row or retrain a head.

## Result

- Mean calibrated-minus-raw NLL: **−0.1943 nats per pair**.
- Paired prompt bootstrap 95% interval: **[−0.2669, −0.1277]**.
- Mean raw NLL: **0.8681**; mean calibrated NLL: **0.6738**; length-only baseline NLL: **0.6931**.
- Per-head mean deltas: **−0.1467, −0.1773, −0.2587**.

This comparison is informative about the already-seen cohort, but it is outcome-informed. It does not repair v2's non-pass or establish that the score scaling will help on new prompts. It also does not show that the reward model improves answer quality, task success, policy learning, or real-world outcomes.

## Planning calculation and limits

Using the v2 prompt-level standard deviation (`0.5654`) in a normal approximation, the standard error at `n=128` is about `0.0500`; an effect near `0.14` nats would be needed for approximately 80% power against a two-sided 95% interval excluding zero. Under the same optimistic approximation, an assumed true effect of `−0.15` has about 85% probability of an upper 95% confidence bound below zero at `n=128`. These figures use a variance estimate from a consumed cohort, omit cohort and head-training uncertainty, and are planning illustrations only—not a preregistered power analysis or a guarantee of replication.

A defensible follow-up should fit each reward head on at least 128 fresh train pairs, fit its positive temperature on at least 128 separate train pairs, and compare raw versus calibrated predictions from that same unchanged head on at least 128 fresh evaluation prompts. Since a positive scalar cannot change margin signs, accuracy is descriptive and should be identical for raw and calibrated scores. A successful follow-up would support only lower pairwise logistic NLL after scalar calibration on the tested HH-RLHF split, model, and sample.

## Reproduction

Run with Python 3.9 and NumPy 1.26.4:

```sh
python3 scripts/analyze_hh_v2_calibration_delta_posthoc.py \
  --output results/cpu-hh-reward-model-v2/posthoc-calibration-vs-raw.json
```

The exact output is retained at [`posthoc-calibration-vs-raw.json`](../results/cpu-hh-reward-model-v2/posthoc-calibration-vs-raw.json). It records the input summary and protocol hashes and labels itself `post_hoc_descriptive_only`.
