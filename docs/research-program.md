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

Decision: CONTINUE for one integration-and-cost feasibility cycle in an existing
resettable multi-turn collector. No additional toy variants or model campaign
from the current checks alone. The next gate must charge donor generation,
historical logits, target prefill, replay/restoration and full group completion;
no net practical savings or no auditable unused-donor lineage stops this attempt.
Keep the independent final task checker separate from evolving feedback.

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
