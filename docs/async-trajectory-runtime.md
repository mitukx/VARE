# Durable asynchronous trajectory execution

Starting point: main `ac54f7a`; PR [#69](https://github.com/mitukx/VARE/pull/69) (continuous real-model updates/SIGKILL recovery) and [#70](https://github.com/mitukx/VARE/pull/70) (research qualification) are **open**, not dependencies silently assumed merged. Existing `CapabilityLoop`, RVL trainers, authenticated reward admission, independent evaluation, `PromotionJournal` and evaluation-job leases remain their owners. The missing boundary is incremental agent-step persistence and its recoverable execute → verify → replay-eligibility handoff.

## Prior art and scope

Inspected on 2026-10-17; public source revisions/hashes are in [the source inventory](../results/trajectory-runtime-v1/prior-art-sources.json). **NE** means not established by these inspected sources, not absent from the whole project. These are source/documentation observations, not independent execution benchmarks. Evaluation hooks do not by themselves establish scientific independence or authenticated promotion.

| Framework | Distributed execution | In-flight updates | Durable trajectories | Replay / carryover | Verifier freshness | Policy provenance | Fault recovery | Independent evaluation | Promotion integrity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [verl Fully Async](https://github.com/volcengine/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/docs/advance/fully_async.md), [Uni-Agent](https://github.com/verl-project/uni-agent/blob/b2bbf83b0ee54fe31d68f4363ddb8aba72fe9913/README.md) | Distributed trainer/agent workers | Partial rollout and weight sync | Rollouter checkpointing; inspected message queue is RAM | Bounded queue/carryover | NE | Trajectory parameter versions | Model/rollouter checkpoints | Validation path; independence NE | NE |
| [slime/straw](https://github.com/THUDM/slime/blob/b2182cc5db977553455eb6e2e5c381eefbb35cfc/docs/en/advanced/straw.md) | Ray control; filesystem tensor transport | Persist partial groups at sync | Filesystem queues and partial groups | Durable batches/recovery pins | NE | Oldest generated-token version | Renewable leases; model/optimizer/RNG recovery journal | Reward/evaluation hooks; independence NE | NE |
| [AReaL](https://github.com/areal-project/AReaL/blob/bb13e3bdb9a865608eafa017a70a790124c3d3bb/areal/infra/staleness_manager.py) | Async inference/training services | Version-aware async admission | Step-boundary durability NE | Capacity/staleness pool | NE | Live version provider | [DCP weights/optimizer/RNG](https://github.com/areal-project/AReaL/blob/bb13e3bdb9a865608eafa017a70a790124c3d3bb/docs/en/reference/checkpointing.md) | Evaluation export; independence NE | NE |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL/blob/2c54cefc7c6b67fef2b51401bcb7eb0e87d8b98d/skyrl/train/fully_async_trainer.py) | Distributed trainer/generators | [Partial multi-turn rollout](https://docs.skyrl.ai/docs/tutorials/fully_async) | Consumed UID checkpoint; durable in-flight steps NE | Async group buffer | NE | Version/staleness statistics | Model checkpoint plus consumed/filtered UIDs | Evaluation path; independence NE | NE |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py) | Ray/vLLM/DeepSpeed | Generation/weight-broadcast lock; partial rollout | Durable step history NE | Ray queue/backpressure | NE | Trainer-step slot; exact per-step checkpoint NE | Training checkpoint path | Separate eval dataloader; attestation NE | NE |
| [Agent Lightning v1.0.0](https://github.com/microsoft/agent-lightning/blob/8f8b8f95fd4ed664822734811b61bb3341f4ddfe/docs/7-asynchronous-training.md) | Local/Kubernetes agents; verl training | Gateway drains requests before update | Inspected [server store](https://github.com/microsoft/agent-lightning/blob/8f8b8f95fd4ed664822734811b61bb3341f4ddfe/agentlightning/server/store.py) is RAM | Intact groups carried over | NE | Proxy model-version metadata | Request retry; durable step recovery NE | NE | NE |
| VARE before this change | CPU/single-host | RVL model lock | Evaluation-job durability; agent recorder closes one trace | Freshness-aware intact group replay | Existing live policy/verifier lag gate | RVL generation/checkpoint contracts | Evaluation leases and promotion journal; #69 separate | Attested checkpoint-bound gate | Explicit publication acknowledgment/recovery fence |

Durable queues, leases, partial continuation, async execution and staleness control are **not novel**. This contribution interoperates at the control-plane boundary; it does not reproduce distributed optimizers. Policy/verifier co-evolution and independent capability improvement remain research objectives, not demonstrated advantages.

## Implemented contract

- `TrajectoryStore` reuses the existing `Store.connection` transactions and hash-chain event writer. Bounded step records atomically append history and advance a resumable JSON cursor. Trajectory/task/group/member identities, worker attempts, fencing generations, policy IDs/versions/checkpoint digests and verifier/config identities are explicit.
- Execute and verify workers make progress independently through durable queues. Heartbeats, bounded retries/lifetime, cancellation, SQLite page limits and producer backpressure constrain resources. Expired workers cannot commit. An unfinished trajectory cannot become complete. Whole trajectories are streamed rather than loaded into memory.
- Tool calls carry a stable operation ID and explicit external execution semantics. A committed step has one effective key; tool execution can occur again after a crash. Serialized state is application state, not an OS/sandbox snapshot.
- Signed rewards bind the complete trace head and verifier result. Terminal-action GRPO admission requires a complete verified group, one identical current checkpoint across **all** steps/members, one terminal model input, current verifier and an explicit reward-attribution rule. Unsupported multi-policy/off-policy/partial cases fail closed. This is an eligibility interface producing existing `Experience` values, **not an optimizer transaction**. The existing live trainer freshness/authentication checks still apply when connected.

The benchmark's synthetic consumer acknowledges eligibility only. It does not call a learner, evaluate a neural policy, or promote a checkpoint. A consumer acknowledgment must never be interpreted as proof of an optimizer commit. Actual loaded-weight identities require a backend check; this generic runtime records declared checkpoint provenance. HMAC keys belong outside agent/tool privilege domains; the local fixture is not a hostile-code sandbox. Public artifact audits verify anchored bytes and known synthetic truth; reward MACs are verified online and the signing key is not published.

## Reproduce

Use Python 3.11+, no GPU or services:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_trajectory_store.py tests/test_trajectory_admission.py tests/test_trajectory_workers.py tests/test_trajectory_benchmark.py
python scripts/benchmark_trajectory_runtime.py --trajectories 100 --mode async --seed 75011 --sigkill --output /tmp/vare-trajectory-screen-new
python scripts/audit_trajectory_runtime.py /tmp/vare-trajectory-screen-new
```

The [fixed systems protocol](../benchmarks/trajectory_runtime_v2/protocol.json) defines matched sync/async modes, three seeds, Pareto step-count/duration tails, Stage A then Stage B, bounded storage/backlog, timing boundaries and abort conditions. New output directories are mandatory. Source snapshots, protocol, raw latency samples, a SQLite snapshot and hashes are retained per run. Prior observed research protocols/cohorts remain unchanged.

For each declared seed, use fresh directories and run both modes with `--trajectories 10000`; audit each output. Do not infer real-model learning, promotion correctness, learner crash recovery or GPU throughput from this workload. Stage D is gated on a separate qualified task/evaluator; no new learning study is authorized by this systems benchmark.
