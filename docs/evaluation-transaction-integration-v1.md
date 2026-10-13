# Evaluation transaction integration: cancellation and revocation

## Problem and contribution

This integrates the existing inference and evaluator lifetime corrections from [PR #19](https://github.com/mitukx/VARE/pull/19) with main `a0f6464cea402ec10a413a704fe092970b3f78a4`, including authenticated training rewards, operator authorization, checkpoint-bound promotion, and the durable promotion journal. Those earlier corrections are reused work, not new algorithms introduced by this integration.

Main's evaluation-error handler directly awaited asynchronous candidate discard. After an operator revoked evaluation authority, another caller cancellation could interrupt that restore, release the round lock, and leave the candidate snapshot retained. Main also used ordinary `gather` for paired evaluation, allowing a failed evaluator's sibling to continue past candidate discard. The latter defect and its correction already have a retained [13-case study](engine-evaluation-lifecycle-v1-report.md).

The integrated order is: verify evaluation authority before scheduling either hook; own and shield both evaluators; verify authority again after they finish; on failure, drain all evaluator tasks and then finish candidate discard in a shielded recovery task before unlocking the round. Repeated caller cancellation cannot abandon that recovery. The original evaluation/authorization error is propagated unless discard fails, which explicitly reports unknown policy state with its cause. Already-cancelling evaluators receive no second cancellation request. Successful paired evaluations remain concurrent.

## Frozen CPU comparison

The [protocol](../protocols/engine_evaluation_interlock_integration_v1.lock.json) was written before running the four-case baseline/candidate comparison. It crosses pre/post evaluation revocation with successful/failing asynchronous discard. Each case sends three caller cancellation requests during discard and queues a competing round. No model download, training cohort, accelerator, or paid API is used.

Local validation commit: `d872933` (not published as a remote Git object). Publication uses the GitHub connector; retained source SHA-256 values identify the tested implementation independently of commit author metadata. The [raw records and manifest](../results/engine-evaluation-interlock-integration-v1/run-1/) bind the exact baseline, candidate, protocol, engine, test, runner, logs, and JUnit hashes.

| Observation | Pinned main | Integrated candidate |
| --- | --- | --- |
| Four revocation/rollback cases | 4 fail; no setup errors/skips | 4 pass; no errors/skips |
| Repeated cancellation during asynchronous discard | Abandons restore and completes the round early | Retains candidate and round lock until discard is terminal |
| Evaluation denied before launch | No hook runs | No hook runs |
| Evaluation denied after completion | Both hooks ran | Both hooks ran |
| Competing round before restore completes | Transaction no longer owns its lock | No new rollout enters |
| Failed discard | Cancellation obscures restore outcome | Explicit unknown-state error; original discard cause retained |

The baseline runner checks that every failure is specifically the rollback-abandonment assertion, rather than a missing dependency or unrelated regression. The candidate's retained new-source hash is authoritative; the short revision above is only a locator.

```bash
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_engine_evaluation_interlock_integration_v1.py --output /tmp/vare-interlock-replay
python scripts/reproduce_engine_evaluation_lifecycle_v1.py --output /tmp/vare-lifecycle-replay
```

The integrated lifecycle replay retains baseline 12 failures/13 cases and candidate 13 passes. See [fresh replay](../results/engine-evaluation-interlock-integration-v1/evaluation-replay/). The separate inference reproduction retains its historical failing arms rather than replacing their outcomes; its [fresh replay](../results/engine-evaluation-interlock-integration-v1/inference-replay/) exercises CPU module modes, RNG, optimizer/model preservation, cancellation, and actual local tiny GPT-2 generation through pinned RVL. No language-model quality metric is measured by these probes.

Local validation: the default full suite passes **600 tests**; the separate mathematical-grader suite passes **14 tests**. Ruff 0.15.12 and package-wide mypy 1.19.1 pass. The [validation record](../results/engine-evaluation-interlock-integration-v1/validation/) retains the logs and their hashes. These local checks do not substitute for GitHub CI.

## Literature and OSS decision

Primary sources checked on 2026-10-14:

| Source | Documented scope | Decision for this contribution |
| --- | --- | --- |
| [verl rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html), updated 2026-09-23 | Importance/rejection sampling for training/inference mismatch, temporal lag, and replay | Reuse trainers; do not claim a novel off-policy correction. |
| [TRL AsyncGRPO](https://huggingface.co/docs/trl/main/en/async_grpo_trainer) | Decoupled rollout/training and staleness-aware asynchronous execution | Do not add another trainer or queue. Validate the outer candidate/evaluation lifetime contract. |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Ray-based RL and asynchronous agent training | Serving and distributed optimizer implementation remain outside VARE's contribution. |
| [vLLM async RL](https://docs.vllm.ai/en/latest/training/async_rl/) and [SGLang RL source documentation](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/sglang_for_rl.mdx) | Serving, generation control, and weight updates for RL integrations | Retain explicit hook resource ownership rather than duplicate serving infrastructure. |
| [Egashira et al., systematic verifier error](https://arxiv.org/abs/2605.02909) | Error structure can shape RLVR outcomes beyond an overall error rate | Signed provenance is not evidence of reward correctness or model improvement. |
| [Moya et al., verifier feedback limits](https://arxiv.org/abs/2609.35677), posted 2026-09-28 | Limits of verifier-only observations and audit-informed selective control | Do not infer clean task success from proxy reward or introduce another generic adaptive-audit method. |

This is a focused implementation decision, not a systematic literature review or proof that other frameworks lack transactional evaluation. VARE's role here is an auditable outer-loop contract layered over a trainer: independently evaluate a candidate, then promote or restore it while retaining the evidence. The measured difference is against pinned VARE main, not against the throughput or quality of those OSS projects.

## Limits and next decision

This closes a bounded in-process correctness gap. It establishes neither algorithmic novelty, real-model held-out improvement, production incidence, distributed correctness, nor recursive improvement. Coroutine termination must mean that each hook has finished its own resource work; detached workers need hook-level ownership. A noncooperative hook can hold cleanup indefinitely. Failed discard reports unknown state and does not prove restoration or add a global recovery quarantine. Successful-path promotion interruption and process-loss recovery remain separate contracts.

No frozen study, consumed confirmation prompt, threshold, or model checkpoint was changed. The remaining research priority is independent task-success evidence after a real update under a viable fresh model/task protocol. This integration does not authorize reopening retired experiments.
