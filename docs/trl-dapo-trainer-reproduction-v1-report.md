# TRL DAPO trainer reproduction v1

**Decision: NON-PASS; stop this line as a standalone contribution.** The frozen
protocol did not pass its full regression matrix. One matched single-window
case reproduced the known normalization difference in the actual trainer; the
multi-window case invalidated the fixture's expected-loss reconstruction after
the policy changed. This is not a new defect report or a capability result.

**Protocol:** [`trl_dapo_trainer_reproduction_v1.lock.json`](../protocols/trl_dapo_trainer_reproduction_v1.lock.json)

**Runner:** [`run_trl_dapo_trainer_reproduction.py`](../scripts/run_trl_dapo_trainer_reproduction.py)

**Raw run:** [`run-1`](../results/trl-dapo-trainer-reproduction-v1/run-1/)

## Question and prior work

Does the merged TRL #6024 DAPO normalization change produce the predicted
loss scale when executed through `GRPOTrainer`, compared with the pinned
pre-fix source? The upstream defect and fix are already documented in
[TRL issue #5619](https://github.com/huggingface/trl/issues/5619) and merged
[PR #6024](https://github.com/huggingface/trl/pull/6024). This study attempted
an independent CPU execution of that known correction. It does not claim a new
algorithm or newly discovered upstream bug.

The run loaded the exact pinned source revisions and the locally cached,
hash-locked DistilGPT2 files. It used CPU-only training, an in-memory repeated
prompt dataset, a synthetic character-length reward, and no network, GPU, or
paid service. Python 3.12.12, PyTorch 2.9.1, and Transformers 4.57.3 were used.

## Frozen estimand and failure

The frozen protocol compared each captured microbatch loss with a reconstruction
from `-advantages * completion_token_counts`, divided by the accumulation-window
token denominator. Its predeclared predictions were baseline `A/S` and fixed
`1`, where `S` is steps per generation and `A` is gradient-accumulation steps.
The protocol required every microbatch in three configurations to meet that
prediction.

That reconstruction is only valid when the captured policy-gradient terms
reduce to the recorded advantages and token counts. In the `S=4, A=2` arms,
the generation schedule spans two optimizer steps. The actual DAPO objective
also includes clipped importance ratios (pinned source lines 3080–3107), and
the denominator correction is at lines 3156–3160. A separate source review
confirmed that v1's expected-loss formula assumed the importance ratio stayed
at one; that assumption is invalid for these runs. The trainer reported
nonzero policy clipping (`clip_ratio/region_mean` 0.2292 and 0.3333), and
optimizer updates changed the policy between portions of the generation
schedule. The baseline observed ratios were 0.435879, 1.708675, 2.752660, and
1.522939, rather than 0.5. The fixed arm's first failing ratio was 0.871758,
rather than 1. The trainer completed both optimizer steps in both arms. The
fixed arm's original runner raised before serializing all captured records; the
partial log and its limitation are retained in
[`fixed-S4-A2.failed.json`](../results/trl-dapo-trainer-reproduction-v1/run-1/fixed-S4-A2.failed.json).

This is a failure of the frozen measurement assumption for this configuration,
not evidence against the upstream fix. The protocol's global acceptance gate
therefore failed; the successful rows below must not be presented as a full
protocol pass.

## Results

| `S` | `A` | Revision | Predicted loss ratio | Captured ratios | Status |
|---:|---:|---|---:|---|---|
| 2 | 2 | baseline | 1 | 1, 1, 1, 1 | captured control |
| 2 | 2 | fixed | 1 | 1, 1, 1, 1 | captured control |
| 2 | 4 | baseline | 2 | 2, 2, 2, 2 | pass for this arm |
| 2 | 4 | fixed | 1 | 1, 1, 1, 1 | pass for this arm |
| 4 | 2 | baseline | 0.5 | 0.435879, 1.708675, 2.752660, 1.522939 | non-pass |
| 4 | 2 | fixed | 1 | first observed 0.871758 | non-pass; partial record |

The equal-window control and `S=2, A=4` pair each changed 76 parameter tensors.
For `S=2, A=4`, both revisions had the same recorded aggregate parameter-delta
statistics (maximum absolute delta `1.1920928955078125e-7`; total absolute
delta `5.532447450670588`). The baseline's logged gradient norm was
`21.8675`; the fixed revision's was `10.9338`, a factor of two, while the
logged clipping ratios were zero. These aggregates are consistent with a
uniform loss-scale change and do not establish an optimizer-update or task
outcome improvement. Final parameter tensors were not retained or compared
elementwise. The separate source reviewer also confirmed the source hashes and
the ratio values from the raw records; this was an internal code/log audit, not
an external reproduction.

Each successful arm took about 25 seconds and peaked at about 2.5 GiB RSS. The
two-step `S=4, A=2` runs took about 30–34 seconds. No accelerator was used.

## Claim boundary and next decision

The study confirms that, in one single-window cached-model execution, the
pre-fix trainer's captured DAPO loss is twice the reconstructed window target,
while the merged fix matches it; the equal-window control is unchanged. It does
not establish that the loss rescaling changes Adam's resulting policy update,
that downstream task success improves, that the correction generalizes across
distributed or mixed-precision execution, or that an external reviewer has
reproduced the result. The reward was synthetic and no independent capability
metric was run.

**Decision: STOP this reproduction line.** The upstream change is already
merged, the full frozen protocol failed, and its positive row does not establish
training-impact value beyond the known normalization behavior. Do not repair the
consumed cohort or loosen its criteria. Return to the repository's higher
priority gap: a concrete, still-open trainer correctness defect with a valid
baseline reproducer, or independent review/reproduction of existing retained
evidence. Preserve this non-pass and do not open another nearby model/config
variant merely to obtain a pass.

## Reproduction

The source and model hashes in the frozen protocol must match. With the same
local checkouts and cached model, run each arm using
[`run_trl_dapo_trainer_reproduction.py`](../scripts/run_trl_dapo_trainer_reproduction.py)
and the arguments recorded in the JSON files under `results/.../run-1/`. Set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`; set
`PYTHONPATH` to the matching pinned TRL checkout. The arm-specific records
retain source/model hashes, runtime versions, captured losses, optimizer-delta
aggregates, and resource use.
