# Interactive rollout reuse: source and public-donor applicability gate

Decision: **STOP this public-donor route**. The broader selected collector
question remains unvalidated; the research objective is not achieved. This is a
bounded source/data audit, not a new method, theorem, capability result or novel
bug report. Do not add neighboring dataset or finite-example screens from it.

## What changed the next action

Two immutable public SWE-smith trajectory records contain genuine interactive
code-editing histories. Neither provides recorded token IDs/probabilities, exact
sampling settings, a recoverable behavior-policy artifact, complete-group
membership or independent target learner/feedback lineage. They therefore do not
supply donors for the selected exact-reuse experiment. Absence here is not a
claim that no suitable data exists anywhere, nor a criticism of their stated SFT
use. See [protocol](protocol.json) and [observations](observations.json).

The data source is pinned to revision
`08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, file
`data/tool-00000-of-00008.parquet`, zero-based rows 1 and 137. Row 1 had already
been previewed and is exploratory; row 137 was fixed before fetching its content.
This is not a representative sample or publicly preregistered confirmation.

| Row | Task/transcript project | Explicit edited file | Stored patch file |
| --- | --- | --- | --- |
| 1 | APISpec | `src/apispec/core.py` | `flashtext/keyword.py` |
| 137 | tenacity | `tenacity/__init__.py` | `src/docx/opc/pkgreader.py` |

The patch mismatch is an **already reported issue**, dating to August 2025:
[dataset discussion #3](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/discussions/3).
We did not execute the patches, independently grade either trajectory, establish
that `resolved` is wrong, identify a training consumer of `patch`, or measure a
learning effect. Do not relabel this reproduction as an empirical discovery.
The [official dataset card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
describes a trajectory fine-tuning use; this audit does not refute that use.

## Pinned source boundaries

[Source observations](source-contracts.json) retain exact commits, files, line
ranges and SHA-256 values. These are source-control-flow observations, not live
distributed executions.

- AReaL `298412a`: its deterministic selector orders **completed** results. Its
  configuration already warns that nonzero head off-policyness does not guarantee
  deterministic task-to-weight-version mapping. This is an acknowledged
  contract, not a newly discovered bug or a demonstrated independence violation.
- CURE `24f37c2`: the published orchestrator waits for sampling, execution, reward
  assignment and training sequentially. In the inspected path, coder reward
  comes from ground-truth tests; this does not demonstrate an asynchronously
  updated tester changing observations inside coder training trajectories.
- RVL `ba94a9e`: the tool agent stores a public grader callback; its terminal
  verifier/reward attachment is separate. The callback can be implemented with
  changing state, but its existence does not demonstrate learned feedback
  co-evolution during useful real-model training.
- [CAPF v1](https://arxiv.org/html/2606.01830) demonstrates verifier-side feedback
  inside training rollouts, while explicitly using a **fixed** feedback generator
  (training setup and Appendix C.2). It supports the observation-drift mechanism;
  it does not establish the incidence of evolving feedback generators.

Completion/length bias and wall-clock selection have existing treatments in
[asynchronous RL](https://www.appliedcompute.com/research/staleness-in-fully-async-rl)
and [Anytime Monte Carlo](https://arxiv.org/abs/1612.03319). Nonstationary replay,
importance weighting and fingerprints are established in
[multiagent RL](https://proceedings.mlr.press/v70/foerster17b.html). None is claimed
as new here.

## Remaining research requirement

The one selected question remains whether distribution-correct reuse of
**unconsumed complete interactive groups** after policy/feedback updates offers
net cost/capability value. The sought contribution is an integrated, useful
collector with measured benefit, not these known probability primitives.

Before another real-model campaign, the integration must establish an actual
resettable feedback path, pinned recoverable old/new sampling distributions,
full-group admission, and target independence including schedule/selection
influence. Do not infer independence merely from “not trained on this donor.”
Current-target history fidelity is a chosen objective; replay in historical
harness contexts can be a legitimate different objective, so compare practical
learning value rather than declaring all historical-context replay incorrect.

Keep the original 32-task pilot, five collector comparisons, total-cost accounting,
15% saving gate, and separate powered 1-point non-inferiority gate unchanged.
Those experiments have not run. No new feasibility pairing is admitted by this
audit, and no stopped model/task cohort is reopened. Additional GPUs alone do not
resolve the importance, integration or novelty gaps.

## Reproduction and resource record

`fetch_records.py` retrieves the exact two records from the pinned Parquet file
using PyArrow/fsspec. Run it after `inspect_schema.py`, then run
`python3 analyze_records.py`. The retained `dataset-info.json`, metadata and
protocol identify the same revision. Use the package versions in
`environment.json`; the local interpreter was Python 3.12.12.

The first download was explicitly interrupted after discovering serial 64KiB
range fetching. Its script and traceback are retained; changing the cache block
size to 4MiB left the record protocol unchanged. The corrected fetch finished
normally. No transcript command or repository code was executed, no model was
called, and no learner update was made.

Raw records and original API responses are retained in the local deliverable
bundle; the public packet includes hashes, extraction code, summaries and URLs.
A hash/record audit is evidence bookkeeping, not a successful transport experiment
or outside reproduction.

A browser resource check found zero active sessions and a visible compute-unit
balance; no runtime was connected, no notebook was saved, and no unit consumption
was tested. The owner subsequently reported an available free GPU session; its
notebook URL/specification is pending. This snapshot does not establish that a
free GPU is unavailable. No paid allocation or live background job exists.
