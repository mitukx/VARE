# ResearchGym export contract

Schema: `vare.research_lab.qualification_export.v1`

`vare research export` writes one JSON object from a run journal. A downstream
trainer may read it later. This package does not call GRPO, does not update an
optimizer, and does not open a held-out split.

Required properties of every export:

- `training_authorized` is `false`.
- `reward_recomputed` is `false`.
- `held_out_split_accessed` is `false`.
- `model_backed` is `false`.
- `counts_as_autonomous_model_success` is `false`.
- There is no `reward`, `advantage`, `return`, or `heldout_ids` field.
- Task ids that start with `heldout`, `run002`, or `coding_policy_learning` are refused.
- A journal metric literally named `reward` is refused.

The export lists actions, public or hidden metric summaries, source hashes,
and the verification bit already stored on a `submit` record. It does not
recompute a learning signal from that bit. A later qualification step, outside
this package, would have to define its own reward and its own split. Nothing
here grants that step.

Eligible inputs are ResearchGym runs whose task id is one of the
`research_lab_v1` pack. Passing verification means the hidden grader accepted
that candidate. It does not mean the trajectory is training data.
