# HH-RLHF reward-model v2

## Question

Can a scalar reward head retain its preference-ranking signal while improving pairwise probability quality, when a single positive temperature is fitted only from out-of-fold training predictions?

## Why this is a new study

Frozen v1 development improved pairwise accuracy by 8.46 points over the response-length baseline but worsened NLL by 0.1635. It failed the combined gate. The new calibration rule is fitted only from out-of-fold training margins; development and confirmation outcomes do not choose its scale.

V1's test indices [0,512) were used for development. V1 never opened its reserved [512,1024) confirmation block because its gate failed. V2 assigns those still-unread indices to a new development cohort and reserves [1024,1536) for confirmation. The v2 lock explicitly excludes the 256 v1 development prompt-context hashes from both new cohorts, in case exact contexts repeat.

## Frozen design

The study reuses the three v1 hash-ranked cohorts of 128 training pairs from the same pinned Anthropic HH-RLHF helpful-base split and frozen Qwen2.5-0.5B-Instruct features. For each cohort, ranks with even positions and odd positions form two folds. Fit a pairwise Bradley–Terry head on one fold and score the other, reversing folds to produce one out-of-fold margin for each of the 128 training pairs. Fit alpha in (0,10) by minimizing mean(softplus(-alpha * margin)) over those out-of-fold margins. The fixed upper bound prevents the unregularized one-parameter likelihood from sending a perfectly ranked training sample toward an arbitrarily large confidence scale. Apply alpha unchanged to evaluation margins from the head then fit on all 128 pairs.

The length-only baseline is trained on all 128 pairs and is not temperature-scaled. Development selects the first 256 eligible unique prompt contexts in official test indices [512,1024). Confirmation indices [1024,1536) remain sealed unless the development audit passes the frozen joint accuracy/NLL gate. Both test cohorts exclude the v1 development context hashes, pilot contexts, training contexts and, for confirmation, v2 development contexts. Exact tokenizer rules, metrics and thresholds are in the [locked protocol](../protocols/cpu_hh_reward_model_v2.lock.json).

The OOF scale is fitted on margins from heads trained on 64 pairs per fold, then transferred to an evaluation head trained on all 128 pairs. This training-size transfer is an assumption. OOF NLL is a training-only fitting diagnostic; it is not an independent calibration estimate for the final head.

The Arrow test cache existed before v1's freeze. Individual v1 dev rows were read and are disclosed; the selected prompt hashes are recorded in the v2 lock. The v2 development and confirmation rows had not been inspected before the v2 lock. The runner hashes the entire compressed test file as opaque bytes for provenance; that operation does not parse row contents. The gzip row reader returns only records inside the selected fixed range and stops before requesting the first line at the exclusive end. Before an audited development pass, neither runner nor auditor may return, parse, tokenize, feature-extract, or score a confirmation row. Gzip libraries may buffer compressed input internally to reach preceding rows; this is not row-level access.

## Evaluation

For each pair, the chosen response is the observed preference. Report pairwise accuracy, logistic NLL, Brier score, ten-bin ECE, paired prompt-level bootstrap intervals, baseline comparisons, the fitted alpha, out-of-fold calibration diagnostics, and CPU resource records. Exact ties receive half credit. The full gate requires mean accuracy above chance, at least a two-point accuracy gain with interval above zero, lower mean NLL with its interval below zero, and at least two of three heads above chance. V1 results motivate the hypothesis but do not alter the frozen v2 threshold.

A pass would support only pairwise preference prediction on this one public source, sample and feature extractor. It would not establish policy improvement, assistant quality, safety, reasoning, truthfulness, generalization, or frontier-scale behavior. No GPU, paid compute, or network access is used.

## Reproduction

After the v2 protocol, runner, auditor, and row-boundary checks are committed, run the development runner and audit. Only after a passing development audit that independently replays the data selection and model scores, run confirmation with its audit record. Preserve a non-pass and keep confirmation closed if the gate fails. The replay uses the same pinned local model and runtime and is not an external reproduction.
