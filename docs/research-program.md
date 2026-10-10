# Research program — 2026-10-11

The objective is to develop technically novel, empirically grounded methods and
systems that improve capability, reliability, or efficiency of frontier-scale
reinforcement learning and autonomous agents. Prioritize long-horizon coding,
tool use, and research-agent post-training. Available hardware constrains what
can be demonstrated today, not which problem matters. Do not invent scaling
measurements or reopen retired cohorts.

The bounded PR #91 same-epoch prepare review is completed at RVL
`ba94a9e513fdaf882cde091766007a88b9edaeee`. The reproduced identity mismatch now
fails closed; all four CI workflows pass. The PR is reviewable and unmerged.
This is a correctness contribution using known fencing/identity principles, not
a new RL algorithm. Stop the surrounding microbug search.

The one selected next question is **distribution-correct reuse of unconsumed
interactive rollout groups after policy and agent-visible verifier feedback
updates**. The prerequisite finite-law and real tensor-trainer checks are
retained in [the packet](../experiments/interactive_rollout_reuse_v1/research-plan.md).
They expose three assumptions: correct residual sampling, donor independence
from both target learners, and feedback replay. They do not demonstrate a novel
method, model gain, or net efficiency improvement.

The selected integration cycle has now executed one frozen local 7B preflight
in the unchanged RVL collector. Decision: **STOP this implementation attempt**.
Cached-decoding versus full-sequence sampled-logprob reconstruction differed by
0.0432244, exceeding the predeclared 0.002 tolerance. Repair stopped before its
first action was admitted. Four episodes on one task produced no edits or public
tests, so feedback repair and useful task outcomes remain untested. See the
[retained packet](../experiments/interactive_collector_preflight_v2/README.md).

This fixed-weight temperature diagnostic used 226.39 seconds of local M1 Pro
process wall time with Metal inference and zero paid compute; it did not measure
pure GPU-seconds, complete repair cost, a learned-checkpoint update or learning
improvement. No threshold/format rescue or rerun of this opened pairing. The
prior 32-task research pilot and its 15% cost/1-point non-inferiority gates are
unchanged and unexecuted. Require materially different, independently justified
feasibility evidence before another model/task screen. Do not substitute more
finite examples or claim a new method from this negative preflight.

The selected scientific question remains open. The largest gap is an integrated
collector that supplies useful real-model learning data with a measured net
cost/capability trade-off. A later attempt must charge donor creation, historical
logits, target prefill, replay/restoration and full group completion; retain a
final task checker independent of evolving training-visible feedback. A failed
local reconstruction tolerance neither proves broad impossibility nor establishes
production incidence or task degradation.

Each cycle: inspect latest evidence; choose one unresolved problem; freeze
success/failure rules; perform one decisive verification; retain code/raw data;
evaluate whether further work is justified. Document closest prior work and
explain importance, novelty, mechanism, falsifiability, currently obtainable
evidence, scale requirements, and external technical utility before a campaign.
A major result must be a nontrivial theoretical contribution, a validated
post-training method, a substantial measured systems contribution, or a
reproducible real-model learning discovery. CI, features, synthetic demonstrations
and training reward alone do not qualify.

The North Star remains open. The current cycle has ended; no ongoing background
training, cloud job, or paid infrastructure is implied by this record.
