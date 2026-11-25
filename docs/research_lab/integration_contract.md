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
that candidate. It does not mean the trajectory is training data, that a
model produced it, or that the same-user worker was an OS sandbox. The
worker does not apply a hard address-space limit or a network namespace.

## Existing OSS task interface

ResearchGym v1's `ResearchTask` is deliberately limited to the three calibrated
CPU experiment tasks in its own catalog. Do not convert an upstream coding task
into one of those task types or let a task manifest supply executable grader
code.

For an existing repository task, use VARE's `EngineeringTaskSpec` as the task
manifest and `AgentTaskContract` as the agent-facing boundary. The manifest
binds the immutable repository revision, evaluator command, protected paths,
resource limits, editable-file metadata, and split label. `load_agent_task`
exposes only the prompt and editable-file allowlist to the coding agent; it
does not expose evaluator files. The task-specific existing grader remains the
owner of source preparation and trusted scoring (for example,
`mlsys_repo_task`, `mlxlm_1912_repo_task`, or `transformers_repo_task`) and must
continue to enforce its own pinned protocol and implementation hashes.

This is an interface and ownership boundary, not a new task qualification or
assignment. A `Development` label does not grant rollout or training
authorization. The frozen model-update gate and task-allocation controls remain
authoritative, and no ResearchGym qualification export is converted into an
RL reward by this contract.
