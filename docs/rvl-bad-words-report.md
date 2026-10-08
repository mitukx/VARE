# Inherited bad-word constraint grader audit

## Question

Can a rollout grader that checks several inherited Hugging Face generation settings still accept a candidate that carries a token-level `bad_words_ids` constraint into sampling?

## Protocol

The added [task](../benchmarks/historical/rvl_bad_words_neutrality/TASK.md) uses the same pinned pre-fix and fixed source revisions as the existing RVL behavior-policy task, with a separate task ID and hash lock. Its CPU fixture gives the model an inherited `bad_words_ids=[[2]]`; token 2 is the first expected response token. The fixture applies the constraint to logits and checks both the effective setting and the resulting rollout probabilities. Existing v4 probability, temperature, state-restoration, and covered generation-setting checks remain active.

The fixture models only single-token bad-word constraints. It does not establish multi-token sequence behavior, compatibility with every Transformers release, or coverage of every `GenerationConfig` field. The grader runs candidate code in a subprocess but is not an OS sandbox.

## Calibration

The frozen calibration grades pre-fix revision `e788f113ad6b246a361cd50a52eaf2867f2a66f8` and fixed revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72`. The pre-fix source failed with maximum absolute rollout and learner log-probability errors of `0.5570354107080964` and `0.37112804442989056`. The fixed source passed with both errors `0.0` across six conditions each. It used no model weights, packages, GPU, paid API, or external compute.

## Mutation comparison

The retained mutation adds `bad_words_ids=[[2]]` to the fixed source's generation configuration. Results:

| Candidate | Existing v4 grader | Added bad-word grader |
| --- | --- | --- |
| Fixed source | Accept | Accept |
| Fixed source plus mutation | Accept | Reject |

This identifies an uncovered setting in the v4 grader's bounded coverage. It is one constructed mutation, not an estimate of false-acceptance rates. The added grader rejects this case because the fixture observes the non-neutral setting and measures its effect on the expected token probabilities.

## Reproduction

```bash
python3 scripts/calibrate_task.py \
  --task-root benchmarks/historical/rvl_bad_words_neutrality \
  --output /tmp/vare-rvl-bad-words-calibration
python3 scripts/audit_rvl_bad_words_mutation.py \
  --output /tmp/vare-rvl-bad-words-mutation
python3 scripts/calibrate_integrity.py \
  --output /tmp/vare-protocol-integrity
```

Raw grades, task snapshots, mutation patch, summaries, and SHA-256 manifests are retained under [`results/rvl-hf-bad-words-neutrality-v1/`](../results/rvl-hf-bad-words-neutrality-v1/). The latest three-task isolated-input integrity calibration is [`cpu-calibration-v5`](../results/protocol-integrity/cpu-calibration-v5/summary.json). Its trust anchor is the unchanged Git-versioned protocol lock; it does not detect coordinated lock edits.
