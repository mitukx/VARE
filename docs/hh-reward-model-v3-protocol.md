# HH-RLHF v3 fixed-head calibration protocol

## Status

**Frozen; development has not run and no v3 test row has been opened.** The machine-readable protocol and lock are [`cpu_hh_reward_model_v3`](../protocols/cpu_hh_reward_model_v3.lock.json). The protocol is frozen before the new test blocks are parsed. Its hypothesis and the `−0.10`-nat practical threshold are informed by a post-hoc analysis of v2's already-opened confirmation cohort; this is a fresh replication of an exploratory signal, not an untouched preregistered claim. The original v2 joint confirmation gate remains a non-pass.

## Question

Does fitting one positive temperature on training-only calibration pairs reduce pairwise logistic NLL on new HH-RLHF prompt contexts when applied to the same unchanged linear reward head, compared with its unscaled margins?

This isolates score-scale calibration from reward-head refitting. Because the scale must be positive, the raw and calibrated margin signs—and therefore pairwise accuracy—must match exactly.

## Frozen procedure

- Use only the already-cached Qwen2.5-0.5B-Instruct revision, the pinned HH-RLHF helpful-base split, and the recorded offline CPU runtime. No downloads, accelerators, or paid services are allowed.
- Exclude every context hash from the retained HH v1 development, v2 development, and v2 confirmation bundles, their selected training contexts, and the six manual pilot contexts. The lock binds those source bundles, the sorted 1,158-hash exclusion inventory, and the exact 768 selected fresh training source-index/context-hash records. The runner and a separate selection implementation agreed on this training-only selection before any new test row was opened.
- Select three fresh disjoint training cohorts of 256 eligible contexts each, ranked by a domain-separated context hash. Fit each linear reward head on the cohort's first 128 contexts; fit its positive scalar temperature on the other 128. Do not refit the head after temperature fitting.
- Development scans the complete fixed test block `[1536,2048)` and retains every eligible unique context after exclusions. It must yield at least 128. Confirmation uses `[2048,2354)`, excludes development contexts, and follows the same all-eligible rule and minimum. No backfill or early stop at 128 is allowed.
- Do not parse confirmation rows unless development passes the frozen gate and the full development bundle passes the independent replay audit.
- The primary estimate averages calibrated-minus-raw NLL across the three fixed heads within each prompt, then averages prompts. Resample prompt contexts—not head/prompt combinations—for the 10,000-replicate percentile interval.
- Pass only when the mean difference is at most `−0.10` nats/pair and the paired 95% interval upper endpoint is below zero. The threshold is a design choice informed by the consumed v2 signal and must be described that way.

## Reporting boundary

A passing confirmation can support only lower pairwise HH logistic NLL after positive scalar calibration on this model, dataset split, and sample. The interval is conditional on the three fitted heads; three heads do not estimate the full training-seed variance. Accuracy is descriptive because scalar temperature cannot change ranking. Brier score and ten-bin ECE are also descriptive, especially at small sample sizes. This experiment does not test downstream task success, policy optimization, assistant quality, generalized reward modeling, or RL efficacy.

The larger unresolved gap remains a fresh post-training run that improves a task-success outcome under independent confirmation. This calibration replication is supporting evidence for careful reward-model measurement only.
