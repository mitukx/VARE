# VARE GRPO group provenance admission audit — v1

**Decision: retain a narrow VARE correctness fix; do not claim a novel RL method or frontier-research contribution.** The frozen CPU reproduction confirmed that individually admissible rollouts from two behavior-policy versions can enter one declared GRPO group and reach the learner hook together. Replay now rejects groups whose members disagree on policy identity/version or verifier version, including legacy mixed groups during grouped sampling. Homogeneous current-policy and uniformly stale-but-permitted groups continue to pass.

## Research question

Does `CapabilityLoop` preserve one behavior-policy provenance inside each declared online GRPO comparison group when `max_policy_lag` admits samples individually? The declared VARE group represents repeated samples requested under one incumbent snapshot. This is an adapter/data-integrity contract for that path. It is not a claim that all GRPO variants must be on-policy or that mixed-policy training is invalid.

The audit protocol was committed before the source-path reproduction in [`vare_group_provenance_homogeneity_v1.lock.json`](../protocols/vare_group_provenance_homogeneity_v1.lock.json), commit `ccb3f791f212e80197fddea4048d3491c3458900`. The bounded repair checks were frozen in [`vare_group_provenance_fix_v2.lock.json`](../protocols/vare_group_provenance_fix_v2.lock.json), commit `1d56f9b03b358fbb37a533c0a6cdd3cfc38bf7d7`.

## Prior art and novelty

This is not a new async-RL provenance concept. TRL's rollout-source RFC already proposes carrying `model_version` on individual samples while preserving a `group_id` for the GRPO comparison set ([TRL issue #5974](https://github.com/huggingface/trl/issues/5974)). Recent work goes further: group-relative REINFORCE has a native off-policy interpretation under the paper's assumptions ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6cad062f78120689e8a678b8af9c6d92-Abstract-Conference.html)), and MiGrATe deliberately forms mixed-policy GRPO groups ([paper](https://arxiv.org/abs/2508.08641)). Thus, policy heterogeneity alone is not a general algorithmic defect.

The project-local gap was narrower: VARE exposed an online rollout-group contract, admitted each member by its own lag, and then used the declared group as the unit for group-relative advantages without checking that the members' behavior-policy identity/version matched. No mixed-policy estimator or explicit mixed-group mode was declared in that VARE path. This is a concrete local correctness repair, but its novelty is low and it does not meet the bar for an original frontier-level research contribution.

## Baseline reproduction

On baseline VARE commit `ccb3f791f212e80197fddea4048d3491c3458900`, the frozen CPU fixture requested active policy `("policy-current", 1)` and returned a two-sample group with policy versions `[1, 0]`. With `max_policy_lag=2`, the engine admitted both rows and called `train_candidate` with both versions in the same group. The retained raw result is [`baseline.json`](../results/vare-group-provenance-homogeneity-v1/baseline.json). It was independently regenerated from the frozen source revision in a detached worktree and matched byte-for-byte.

| Case | Baseline result |
|---|---|
| Mixed policy versions `[1,0]` | Both rows reached `train_candidate` together |
| Homogeneous current `[1,1]` | Both admitted |
| Homogeneous permitted-stale `[0,0]` | Both admitted |
| Future `[2,2]` | Both rejected by the existing lag gate |

The reproduction also attempted a mixed-verifier case. `VerifierEnsemble` canonicalized both outputs to `active_version=1`, so that fixture did not produce mixed verifier provenance at the learner boundary. It is retained as a failed control in `baseline.json` and is not counted as confirmation. A separate replay-level regression now checks verifier-version homogeneity on direct/legacy replay input.

## Exact update contrast

An independent, pure-Python `Fraction` oracle enumerates the four possible two-action groups without importing VARE, PyTorch, or a trainer. The target policy is Bernoulli with `pi(a=1)=1/2`, reward is `r(a)=a`, and the group size is two. For unequal actions, population-standardized group advantages are `(+1,-1)` or `(-1,+1)`; for equal actions they are zero. The update uses the target-policy score `a-pi` and averages over the two members.

With both samples drawn from the target policy, the expected score update is `1/4`. With one group member drawn from Bernoulli(0.9) and the other from Bernoulli(0.1), the actions disagree with probability `0.82`, so the same uncorrected target-score calculation has expected update `41/100`. That is an absolute difference of `4/25` (64% relative to the specified target expectation). Both means point in the same direction; this fixture measures magnitude shift, not sign reversal.

For one group, the exact update variances are `1/16` for target-policy sampling and `369/10000` for mixed behavior. For a mean over two independent groups, MSE relative to the target-policy expectation is `1/32` and `881/20000`, respectively (ratio `881/625`, approximately 1.41). This two-group illustration shows why variance alone can be misleading when provenance changes the estimand. It is a deliberately small synthetic calculation; it does not establish practical prevalence, sample complexity, actual optimizer behavior, or model capability. The complete independent output is [`exact-oracle.json`](../results/vare-group-provenance-homogeneity-v1/exact-oracle.json), generated by [`check_group_policy_provenance_oracle_v1.py`](../scripts/check_group_policy_provenance_oracle_v1.py).

## Fix and checks

The root cause was that replay's group-completeness check compared prompt identity and declared group size, but not policy identity/version or verifier version. The fix adds a shared provenance predicate to `PrioritizedReplay`: atomic admission now rejects heterogeneous groups, and grouped samplers discard legacy groups that fail the same predicate. `CapabilityLoop` emits `heterogeneous_provenance` when it rejects such a complete group.

The source and regression-test diff is retained as [`vare-group-provenance-fix.patch`](../results/vare-group-provenance-homogeneity-v1/vare-group-provenance-fix.patch). It has not been submitted to another project; VARE owns the affected code.

Focused regressions cover the full engine path, current and permitted-stale controls, future-version rejection, atomic rejection of mixed policy IDs/versions and verifier versions, and fail-closed sampling of legacy mixed-policy groups. The focused engine/replay/RVL GRPO set passed **32 tests**; the log is [`focused-tests.log`](../results/vare-group-provenance-homogeneity-v1/focused-tests.log).

The full suite completed with **225 passed, 12 skipped, and four failures** in the pre-existing `smoke-stable-logsumexp` environment campaign/catalog tests. Those same failures are already recorded as reproducible on baseline in [`engine-rollout-task-binding-v1-report.md`](engine-rollout-task-binding-v1-report.md). Full output is retained in [`full-tests.log`](../results/vare-group-provenance-homogeneity-v1/full-tests.log). No new failure in the touched GRPO/replay paths appeared.

## Limitations and decision

- This finding is local to VARE's declared online group contract. It is not a defect report against TRL or RVL and does not establish that the default `RVLGRPOHooks` path produces mixed-policy groups; that adapter restores the requested active snapshot for each rollout.
- The exact gradient result uses a synthetic Bernoulli policy and unclipped score-function update. No real RVL optimizer step or model was run for this audit.
- No task-success, held-out capability, throughput, production prevalence, upstream acceptance, or outside reproduction is established.
- Mixed-policy group methods can be principled when their data construction and target objective are explicit. This patch fails closed for VARE groups whose metadata disagree; it does not prohibit an explicitly designed mixed-policy algorithm.

**Decision: STOP this line as an original research claim; retain the local correctness fix.** The defect and remediation are validated at E0 (harness/data-contract correctness). The result is not a substantial original contribution suitable to present as frontier-lab research impact. Next work should identify a separately novel trainer correctness failure or an affordable independent task-success result, not extend this provenance patch into a new algorithm.

## Reproduction commands

Run post-fix checks from the VARE root with Python 3.12:

```bash
PYTHONPATH=src .venv/bin/python3.12 -m pytest -q \
  tests/test_grouped_policy_provenance.py \
  tests/test_replay_group_prompt_identity.py \
  tests/test_grouped_replay.py \
  tests/test_replay_group_freshness.py \
  tests/test_rvl_grpo_group_boundary.py \
  tests/test_rvl_grpo_group_boundary_calibration.py \
  tests/test_rvl_grpo_hooks.py
PYTHONPATH=src .venv/bin/python3.12 scripts/check_group_policy_provenance_oracle_v1.py
```

To reproduce the pre-fix source path, create a detached worktree at `ccb3f791f212e80197fddea4048d3491c3458900`, copy `scripts/reproduce_group_provenance_baseline_v1.py` into its `scripts/` directory, and run it with `PYTHONPATH=<worktree>/src` and Python 3.12. The result should match `results/vare-group-provenance-homogeneity-v1/baseline.json` byte-for-byte.
