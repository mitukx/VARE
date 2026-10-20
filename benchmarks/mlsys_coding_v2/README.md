# MLSys coding pack v2 (draft)

Draft task notes for review. This directory is not an authorized evaluation
and it has not been used for a held-out policy run.

Train task files stay in `benchmarks/mlsys_coding_v1`. Held-out membership is
not frozen. The comparison draft targets 15-20 held-out tasks, and the
current v1 held-out pool has 4, so membership waits on a 20-30 task
expansion. The selected ids will be frozen by hash before any training.

`mlsys-v1-causal-attention` stays in the v1 split. It is excluded from a
future held-out set because it shares the stable-softmax repair with train
`mlsys-v1-row-log-softmax`. The three utilities here are unassigned pool
candidates, not that held-out set.

The new tasks use the v1 harness adapter. `metadata.task_type` is
`correctness` and is read from this directory's `task.json`, not from the
candidate. Fixture calibration of the baseline and oracle files is a harness
check. It is not a held-out policy score.

`power.py` is the paired-t simulation cited by
`protocols/mlsys_coding_policy_comparison_v2.json`. Run it as a script to
reproduce the minimum-detectable-effect table. Importing it does not draw
random numbers.
