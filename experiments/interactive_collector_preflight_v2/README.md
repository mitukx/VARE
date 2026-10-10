# Interactive collector preflight v2 — stopped

Decision: **STOP this frozen local implementation attempt.** The selected
research question remains unvalidated. Do not tune or rerun this model/task pair,
relax the tolerance, substitute a toy result, or report a speedup from unfinished
repair. The broader research program remains open.

## Question and boundary

Can an asynchronous collector reuse complete, unconsumed interactive rollout
groups after policy and agent-visible verifier feedback change, while preserving
useful current-policy learning signals and reducing total cost?

The proposed contribution is a collector with a joint policy/verifier ancestry
boundary, residual token correction, and replay to the first changed observation.
The known speculative-sampling primitive is not a new algorithm. Previous finite
checks are necessary conditions, not a demonstrated method or systems impact.

This packet executes one engineering preflight in **unchanged RVL
`CodingToolAgent`, `ServingLease`, and `AsyncRolloutEngine`**, pinned at
`ba94a9e513fdaf882cde091766007a88b9edaeee`. Only an experiment-local MLX backend,
temperature-pinning lease adapter and resettable grader are supplied. There is
no learner, optimizer, deployed asynchronous queue or production ancestry ledger.
The donor receipt describes fixed targets with empty training ancestry; it does
not certify a production lineage service.

The cached Mistral-7B-Instruct-v0.3 4-bit model runs on M1 Pro Metal. Donor sampling
temperature is 1; target temperature is .8, with identical weights. This tests a
change of sampling distribution, **not a learned checkpoint update**. One pinned
HumanEval/6 task has two preassigned donor and two fresh members. Neither reward
nor completion duration selects members. Weak public feedback checks the first
upstream assertion; strong feedback checks all three. A separate checker uses
64 balanced-parenthesis cases generated before inference. The task and pretrained
model may overlap in training; no generalization or capability claim follows.

The frozen limits are six action turns, 256 tokens per turn, 600 seconds total,
and zero paid calls. The original 32-task/4-member research pilot, >=15% total
collector GPU-second improvement and <=1-point task non-inferiority gate were
not run or changed by this diagnostic.

## Observed result

The only actual model run terminated with exit code 1 at `repair-0`, turn 0.
Full-sequence old-policy reconstruction disagreed with stored cached-decoding
sampled log-probabilities by **0.04322438595197653**, exceeding the frozen
**0.002** maximum absolute error. The repair group was not admitted and no
learner was called. There was no threshold change or inference rerun.

Both evaluations used the same weights, token IDs and donor temperature. The
two paths are cached autoregressive inference versus a full-sequence forward.
This records a numerical reconstruction incompatibility in this implementation;
it does not identify the responsible kernel, establish a general error bound,
or prove task degradation. It does not falsify residual coupling with the correct
proposal law. Full logit vectors were not retained; recomputing this diagnostic
requires the pinned model/runtime, rather than an offline JSON-only audit.

The finite-vector algebra check returned zero error, but the actual transport
acceptance loop was never reached. That check therefore supplies no empirical
evidence of target-law fidelity. `mean_token_tv=.0232253` compares reconstructed
temperature-1 and temperature-.8 laws; it is **not** reconstruction error TV.

Across the four returned episodes there were 22 actions: 21 invalid JSON actions
and one valid `finish`. There were no edits and no public tests. All four final
sources were empty and received hidden reward 0. This is four episodes on one
opened task, not four independent test tasks. The feedback-disagreement branch,
state restoration after a changed tool observation, successful repair completion
and group-relative learning remain untested. No format/prompt rescue is allowed.

## Measured cost and its limits

| Work | Synchronized host/Metal wall seconds |
| --- | ---: |
| Two donor episodes, including invalid actions | 83.456881 |
| Two fresh episodes, including invalid actions | 135.765292 |
| First failed q/p reconstruction operation only | 1.920305 |
| Model load | 0.916886 |
| Entire measured process, including hashing/checkers/unfinished work | 226.385720 |

No complete repair-arm cost exists. In particular, 1.92 versus 135.77 seconds
is not a speedup comparison. Reconstruction materialized 134,217,728 host bytes
for q/p arrays at the first 256-token action. This same-weight diagnostic shares
one forward between the two temperatures and understates the work of two
different checkpoint reconstructions. Model load and all failed actions remain
in the record. Wall time includes CPU dispatch and synchronization and is not
pure accelerator time or CUDA GPU-seconds. Paid compute/API cost: zero.

For a later complete asynchronous comparison, let D be donor creation, F fresh
collection and R repair, with all associated costs included. Compare D+R to D+F
when both arms incur the same in-flight donor work; also report D+R versus F from
scratch. A 15% end-to-end saving requires R <= .85F-.15D, not merely R <= .85F.
The current incomplete repair supplies no value of R and cannot pass that gate.

## Prior work and remaining contribution

[SPEC-RL](https://arxiv.org/abs/2509.23232) already reuses cached policy prefixes.
[Speculative sampling](https://arxiv.org/abs/2302.01318) supplies residual
correction. [RollVerify](https://arxiv.org/abs/2610.09914) already repairs stale
rollout suffixes. [ReSpec](https://arxiv.org/abs/2510.26475) is a substantial
RL-serving baseline. None of the generic components can be claimed as novel
here. A possible contribution would require an integrated ancestry/feedback
contract plus a demonstrated cost/capability advantage over these baselines.
That contribution has not been established.

A technical reviewer should criticize the fixed-weight temperature perturbation,
single old instruction model/task, failed action format, lack of full-vector
reconstruction evidence, absent feedback replay, and lack of asynchronous
batching or real training. These observations justify stopping this local
implementation attempt; they do not justify a frontier-scale conclusion.

The next large experiment remains the prior plan: a capable coder, genuinely
distinct learned snapshots, 32 fresh tasks x four members, two feedback versions
and gaps 0/1/3 before an adequately powered untouched evaluation. Profile before
allocation; the prospective 2 x 80GB GPUs and 16 GPU-hour cap are a resource
envelope, not measured requirements or an available/started job. No scale claim
or new model screen is authorized by this failed preflight alone.

## Evidence and verification

`protocol.json` and its hash precede inference locally; they are not a public
preregistration. `preflight.py` is the exact executed runner, with SHA recorded in
`run-1/raw.json`. The raw record, four agent journals, checker output, independent
checker cases, donor receipt, model/config fingerprints and package freeze are
retained. `compile-attempt-1.log` records a missing bracket caught before any
inference; the syntax repair changed no protocol or result criterion.

Run the portable, inference-free record audit:

```bash
python3 audit.py .
```

All 74 record-consistency checks pass. They verify metadata/hashes and retained
records; they do **not** turn the failed experiment into a pass, rerun the model,
prove the sampling law or count as outside reproduction. The manifest covers
all packet files except itself.

Original inference invocation, requiring the recorded Apple Silicon environment,
cached model and host-specific paths embedded in the runner:

```bash
work/mlx-env/bin/python work/collector-v2/preflight.py \
  --rvl work/RVL-shallow \
  --protocol-dir work/collector-v2/frozen-not-executed \
  --out work/collector-v2/run-1
```

That output directory now exists and the runner refuses to overwrite it. The
execution is retained as a failed observation; no second run is implied. Full
inference portability and outside reproduction remain limitations. HumanEval
source and license are recorded in `task-source.json` and `HUMANEVAL-LICENSE`.
The original process is terminal; no background training or cloud job remains.
