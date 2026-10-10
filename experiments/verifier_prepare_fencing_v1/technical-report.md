# Verifier prepare commit fencing across asynchronous artifact loads

Date: 2026-10-11. Decision: **CONTINUE for narrow upstream correctness review**. This is a source-level defect and patch, not a new RL algorithm or a model-capability result.

## Problem and selected claim

A verifier deployment needs to bind a reward to the artifact the deployment intended, not merely to an integer version. In the inspected RVL worker, `prepare_verifier` checks a coordinator epoch and active version, then awaits an application-provided artifact loader. While it waits, another request can advance the epoch, prepare a different artifact, or activate a newer version. The old continuation then publishes prepared state without repeating its checks.

The falsifiable implementation claim is: **an obsolete prepare can contaminate a newer deployment, including successful activation of the wrong same-version artifact; revalidation after the loader returns prevents the demonstrated interleavings in one worker event loop.** The selected work is a concrete verifier-refresh correctness study within verification-aware asynchronous RL. It does not reopen the retired partial-label audit or stale-parameter-gradient directions.

Inspected revisions:

- VARE main: `b3204926b3cae291f7c13e3134444c71a3a63172` (also the handoff revision).
- RVL main: `27ebf7fe239d97504eeb960bf1407de220584dac`.
- RVL patch commit: `436a96e86b704ad981b05237b41fdfb846f18817`.

The implementation delta is 11 added source lines and one removed source line, plus seven regression tests and deployment documentation. The existing RPC API and trainer are retained.

## Existing evidence and scope choice

VARE's charter, README, roadmap, current gaps, Stage 0 decision, stopped audit-cost line, stale-gradient applicability report, GRPO adapter, engine and lag controller were reviewed. The latest model evidence is negative and its consumed cohorts remain retired. RVL's README, verifier implementation, artifact deployment coordinator, TCP worker/client, group-advantage helper, HF trainer, and deployment/RPC/routing tests were inspected.

The generic ideas of policy-lag correction, verifier co-evolution, reward relabeling, and audit allocation already have close precedents. This contribution instead targets a reproducible source-level integrity failure. The deployed TCP service is a reusable RVL component; the inspected repository references its fleets in routing benchmarks. **This report does not establish that the default HF training campaign uses this deployment service.** A separate controlled integration explicitly connects its returned rewards to the existing HF trainer.

## Counterexample through public TCP interfaces

Use two immutable artifacts A and B with different SHA-256 values, both at version 1, in separate artifact directories. The coordinator takeover allows the new coordinator to choose B before version 1 has ever become active.

1. Epoch 1 sends prepare(A). Its asynchronous loader starts and waits on an event.
2. Epoch 2 sends prepare(B). B loads, persists and ACKs successfully.
3. Release A's loader. The baseline worker overwrites prepared B with A and returns an ACK carrying its now-current epoch 2.
4. Epoch 2 calls activate(version=1). The baseline activates A.
5. The coordinator's existing convergence check sees version 1 / epoch 2 and returns success, although the active SHA-256 and reward belong to A.

No forged RPC, private state mutation, timing sleep, pretrained model, or GPU is required. An independent public `ping` exposes the incorrect digest; a normal verification RPC exposes the wrong reward. Worker restart reloads the wrong persisted artifact, so this is not only transient in-memory contamination.

Different-version takeover also produces an activation failure; a same-epoch active-version advance allows the baseline to regress from active version 2 to 1. These are separate manifestations of the same missing post-load validation.

## Correctness contract and fix

Let a prepare request carry epoch `e` and manifest `m=(version,path,digest)`. At prepared-state publication, require:

1. `e` still equals the worker's current coordinator epoch.
2. `m.version >= active.version`.
3. If the versions are equal, `m` equals the active manifest; reuse the active verifier for idempotence.
4. Validate before changing either prepared pointer or persisting prepared state.

The patch loads into a local variable, calls the existing epoch fence again after the await, repeats active-version/identity checks, then publishes. No await occurs between these checks and publication. In the worker's single-event-loop execution model, no other handler can interleave at that commit boundary. This is a local reasoning argument, not a formal proof over threads, processes, filesystem failure or a distributed fleet.

The fix uses established fencing and compare-before-commit principles. Chubby is relevant prior art for stale-holder fencing; this work does **not** claim that primitive is new. Its concrete contribution is a previously unreported defect in the pinned RVL source, a minimal correction, and externally reproducible failure schedules. The proposed fix is [RVL draft PR #91](https://github.com/mitukx/Recursive-Verification-Lag/pull/91). A repository issue/PR scan found no matching existing report at inspection time; that scan is not a universal novelty proof.

## Frozen primary evaluation

`protocol.json` was frozen locally and hashed before execution. It was not publicly committed before observation; the retained hash alone does not independently prove chronology. Fixtures were source-informed; they are deterministic correctness probes, not a randomly sampled workload. One full run per revision was specified. All outcomes are retained.

| Event-controlled case | Baseline failed assertions | Patched failed assertions |
|---|---:|---:|
| Epoch takeover, same version / distinct artifact | 4 | 0 |
| Epoch takeover, distinct versions | 6 | 0 |
| Active version advances during load in same epoch | 6 | 0 |
| Non-racing asynchronous loader | 0 | 0 |
| Restart after same-version takeover | 5 | 0 |
| Total | 21 | 0 |

Four of five baseline cases fail; all five patched cases pass. These assertions are correlated and the cases are constructed: **21 is not an independent event count, and 4/5 is not a production failure-rate estimate.** Runtime was 1.224 s for the initial baseline and 0.803 s for the initial patched run, including imports and local service setup. These are descriptive timings; no speedup claim is made.

Code review then added two separately declared controls for equal active versions: conflicting manifests must reject, identical manifests must remain idempotent. Baseline fails the conflict control and passes idempotence. The patched seven-test file plus thirteen existing deployment/RPC/routing tests passes **22/22**. Initial five-case logs, subsequent controls, and the final suite are separate artifacts.

## Controlled connection to the actual GRPO trainer

A supplemental protocol was frozen **after the primary deployment result**, before its own model-update result. It is not independent confirmation. Artifacts return rewards for two response IDs:

- A: `[0.1, 0.9]`.
- Intended B: `[0.9, 0.1]`.

A two-outcome tensor policy has logits `[theta,-theta]` and starts at `theta=0`. The two complete one-token response records store exact behavior log-probability `-log(2)`. Normal TCP verification produces `VerifiedGeneration` objects, which go directly to the **unchanged** `HFCausalLMGRPOTrainer.train_step`, its group-normalization helper, token objective, backward pass, and one SGD update at learning rate 0.1. SGD is explicitly substituted to make the parameter update transparent; this does not measure the trainer's default AdamW behavior.

For rewards `(r0,r1)`, the analytic loss derivative at zero is

`dL/dtheta = -(r0-r1) / (2 * sqrt(((r0-r1)/2)^2 + 1e-6))`.

| Quantity | Baseline | Patched |
|---|---:|---:|
| Actual RPC rewards | `[0.1,0.9]` | `[0.9,0.1]` |
| Loss gradient | +0.9999969006 | -0.9999969006 |
| Gradient error to intended B | 1.9999937756 | 2.55e-8 |
| Theta after SGD | -0.0999996886 | +0.0999996886 |
| Probability of response 0 after update | 0.4501661 | 0.5498338 |
| Maximum absolute behavior/current log ratio | 0 | 0 |
| Clip fraction / behavior-KL diagnostic | 0 / 0 | 0 / 0 |
| Mean reward / reported loss | 0.5 / 0 | 0.5 / 0 |

Both arms agree with the analytic derivative of the rewards they actually received. The trainer is not defective here; the baseline deployment supplied the wrong objective. Common policy-drift, mean-reward, loss and gradient-norm summaries match across arms while the gradient direction reverses. This demonstrates a controlled route from artifact mismatch to optimizer mutation, without claiming that a pretrained model learned a task or that an actual long-horizon run suffered this failure.

## Related work and concrete difference

- [IMPALA](https://arxiv.org/abs/1802.01561) and [AReaL](https://arxiv.org/abs/2505.24298): actor/learner decoupling and off-policy correction. Our counterexample has zero policy mismatch; changing likelihood ratios cannot authenticate the reward artifact.
- [NeMo-RL Async GRPO](https://docs.nvidia.com/nemo/rl/latest/guides/async-grpo.html): policy-lag handling and importance correction. Our finding is at verifier deployment publication, upstream of loss evaluation.
- [PEBBLE](https://arxiv.org/abs/2106.05091): relabeling stored experience when the reward model changes. This assumes the intended scorer can be identified; the counterexample violates that prerequisite.
- [RL Tango](https://arxiv.org/abs/2505.15034) and [Cooper](https://arxiv.org/abs/2508.05613): generator/verifier or reward-model co-evolution. This work supplies a deployment contract and regression, not a co-training algorithm.
- [Scaling Laws for Collapse in Asynchronous GRPO](https://arxiv.org/abs/2607.01083), [Mu-GRPO](https://arxiv.org/abs/2605.17570), and [PNPO](https://arxiv.org/abs/2608.01418): adjacent policy-lag/reuse work; these make a generic staleness method claim unsuitable here.
- [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) and [DREAM-RL](https://openreview.net/pdf?id=9idmRtG6M5): verifier reliability and evolving reward systems. We address whether a published artifact identity remains true through an asynchronous prepare, rather than semantic error or reward-system design.
- [Chubby](https://research.google/pubs/the-chubby-lock-service-for-loosely-coupled-distributed-systems/): established distributed fencing principles. No new consistency primitive is claimed.

This was a targeted primary-source check, not an exhaustive systematic review. Paper-level novelty is **not** established.

## Scaling, measurement and falsification

At large scale, asynchronous checkpoint loading, coordinator takeover, and independent verifier workers create opportunities for this interleaving. That is a systems inference, not a claim about any lab's private architecture. The bug needs concurrency and an await, not a large model; local execution can decide its correctness. Expensive training is unnecessary to justify this patch.

The post-load validation has constant-count integer/manifest comparisons per prepare and introduces no new await or global barrier. The demonstrated patch does not need to hold an epoch lock across checkpoint loading. Large-checkpoint memory coexistence, stale-load cancellation, cleanup, retries and allocator pressure remain unmeasured.

For a real distributed integration, first pin a maintained actor/learner/verifier stack that actually supports refresh and takeover. Run the same invariant checks with real checkpoint loading, network delays, worker restarts and complete rollout groups. On an 8- or 64-GPU cluster, choose a model/parallelism layout that fits measured memory; do not assume this packet predicts hardware throughput. Record intended and actual artifact digest, epoch, policy version, verifier version, group membership, queue age, load/activate latency, rejected old requests, retried verification calls, wasted verification tokens, memory and learner starvation. Compare the pinned baseline and patch under matched predeclared fault schedules. Primary systems outcome: wrongly accepted artifact/reward records; secondary outcomes: recovery cost and time. Independent task success after real training would be a separate, fresh study with no-update and matched learning controls.

Fleet-wide atomic activation is not provided. Other unresolved contracts include durable coordinator consensus, atomic publication across processes, path-to-loader integrity under concurrent file mutation, verifier nondeterminism, and consistent reward epochs within a GRPO group. Version numbers alone are insufficient if multiple artifacts can share a version across coordinator lifetimes; a maintained deployment registry may instead impose global content/version uniqueness, preventing this specific same-version schedule. That is an alternative contract, not assumed in RVL.

Falsification/stop rules:

- If outside reproduction cannot produce the baseline public-API mismatch under the specified schedule, retract the defect claim.
- If a supported interleaving makes the patched prepare publish under an obsolete epoch or older active version, reject the patch as complete.
- If a real target stack forbids asynchronous loading or guarantees globally unique immutable version-to-digest mapping plus correct takeover serialization, do not claim this specific failure applies there.
- If representative large-stack runs never encounter overlapping deployment/takeover and show no measured benefit, keep the local correctness fix but stop broader impact claims.

## Evidence and reproduction

Use Python 3.12 with PyTorch 2.9.1 for the trainer connection. The initial host also has Transformers 4.57.3, although the tiny policy does not instantiate Transformers. Primary deployment tests use standard-library networking and existing RVL code. No paid API, GPU, MPS compute, pretrained checkpoint, or archived benchmark cohort was used.

From this packet directory:

```bash
PYTHON=python3.12 bash reproduce.sh /path/to/new/scratch-directory
python3.12 audit.py
```

The shell script fetches the exact base SHA, applies the retained patch to a separate clone, executes both primary arms and both trainer arms, and runs the 22-test suite. `audit.py` checks the retained local packet; it does not independently rerun TCP schedules. A clean-fetch execution of the shell script reproduced all declared outcomes. Its log is retained. The initial auditor reconstructs 47 checks without importing the runner. This is same-author, same-host scripted verification, not outside human review.

## Final technical assessment

The result is useful as a reusable fault schedule, artifact identity contract and minimal upstream patch. It is technically narrower and better supported than a generic new verification-aware RL algorithm claim. No new theorem, empirical scaling law, trained-model gain, or broad post-training efficiency improvement has been demonstrated. The appropriate next action is maintainer/human review and application to a real refresh-capable stack; a large GPU experiment is not a prerequisite for merging this narrow correctness fix.
