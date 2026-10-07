# VARE — Verification-Aware Reinforcement Engine

**Closed-loop capability improvement for post-training, long-horizon agents, and research automation.**

VARE is an experimental outer control plane that turns model failures into new training pressure while keeping optimization grounded by fresh verification, immutable provenance, and independent held-out promotion gates.

> **For contributors and AI agents:** read [`AGENTS.md`](AGENTS.md) first. It defines the project mission, hard invariants, evidence policy, compute discipline, and how to choose the next task.

> **Status: research prototype / measured CPU control-plane and environment-harness evidence.** The repository has a deterministic CPU end-to-end result and tested integration contracts. The real RVL/Qwen GRPO path is implemented but not yet claimed as GPU-validated capability evidence.

## Core loop

```text
failure discovery -> adaptive curriculum -> grouped async rollouts
 -> verifier ensemble -> policy/verifier freshness + shift gate
 -> group-preserving prioritized replay -> candidate training
 -> independent paired held-out evaluation -> promote or rollback -> repeat
```

North-star metric: **held-out capability gain per unit of scarce compute**, with GPU-hours reported when a learner actually uses accelerators.

## Why separate from Recursive-Verification-Lag?

`mitukx/Recursive-Verification-Lag` already owns substantial lower-level RL infrastructure: token-exact rollout, GRPO, verifier execution, durable replay, verification debt, weight synchronization and GPU harnesses. VARE owns the **outer improvement policy**: what failures to attack, what data to prioritize, when verification debt requires intervention, and whether a candidate deserves promotion.

## Implemented

- bounded async rollout control with unique per-rollout provenance steps;
- explicit `samples_per_task` and **group-preserving replay** for GRPO-style objectives;
- weighted verifier ensemble, disagreement telemetry and trusted-verifier pass/fail ownership;
- policy-lag, verifier-lag and task-distribution-shift gates;
- freshness-aware prioritized replay and failure-driven curriculum;
- failure-driven task generation and verifier refresh/co-evolution hooks;
- paired held-out evaluation support with optional deterministic bootstrap lower-confidence promotion gate;
- fail-closed slice/cost/disagreement promotion constraints;
- tamper-evident promotion ledger and SHA-256 experiment protocol locks;
- read-only RVL `TokenReplay` SQLite inspection and outer-loop planning;
- transactional `RVLGRPOHooks` over RVL `HFLocalBackend + HFCausalLMGRPOTrainer`;
- HTTP chat-completions rollout adapter;
- executable repository-environment factory with task catalogs, isolated workspaces, evaluator integrity checks, CPU resource limits and performance gates;
- failure-matched engineering curriculum and workspace-agent rollout adapter;
- fixed-budget environment campaign runner with hash-linked raw records;
- deterministic CPU reference experiment and CI tests.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . pytest
pytest
vare demo --rounds 8 --rollouts 512 --seed 7 --output artifacts/demo.json
```

The committed L0 seed-7 run improves trusted held-out accuracy from **0.65625 to 0.96680**, with 3 promotions and 5 rejected candidates. This validates orchestration only; it is **not** evidence of language-model self-improvement.

## Compute-constrained path

VARE deliberately separates **environment/verifier research** from expensive learner scale. The repository can make progress with CPU-only historical-task construction, trusted executable grading, failure mining, trajectory analysis and curriculum experiments; accelerators are reserved for experiments that require parameter updates. See [`docs/roadmap.md`](docs/roadmap.md) and [`docs/environment_factory.md`](docs/environment_factory.md).

Smoke-test the engineering environment layer:

```bash
vare env-smoke --output-dir artifacts/env-smoke
```

The committed smoke task is synthetic and validates the harness only. Real coding/engineering claims require immutable historical repository tasks with independent evaluators. `benchmarks/historical/rvl_behavior_policy_parity` is the first such task seed: it references an immutable pre-fix revision and keeps its narrow regression evaluator outside the candidate checkout. It is not yet counted as measured agent evidence.

Run the same task catalog against an external repository-editing agent command:

```bash
vare env-campaign \
  --catalog-root benchmarks/smoke \
  --agent-argv-json '["python","examples/oracle_smoke_agent.py","{workspace}"]' \
  --repeats 2 \
  --run-root artifacts/env-campaign
```

The example agent is an oracle plumbing check only. Replace it with the coding-agent command under evaluation for real trajectory evidence.

## RVL integration

Inspect a live RVL replay database without mutating it:

```bash
vare rvl-plan \
  --replay-sqlite /path/to/replay.sqlite \
  --current-policy-version 12 \
  --current-verifier-version 7 \
  --max-policy-lag 2 \
  --max-verifier-lag 1 \
  --output artifacts/outer-plan.json
```

Run the transactional real-model path from an environment that already has RVL's pinned GPU dependencies:

```bash
PYTHONPATH=/path/to/Recursive-Verification-Lag:/path/to/VARE/src \
python examples/run_rvl_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset gsm8k \
  --rounds 8 --prompts-per-round 8 --samples-per-prompt 8
```

See `docs/rvl_outer_loop.md`. The multi-seed L2 experiment is preregistered in `protocols/l2_rvl_qwen_v1.lock.json`; the lock SHA is part of the evidence chain.

## Evidence ladder

- **L0:** deterministic CPU control-plane experiment — completed.
- **L1:** tiny causal-LM import/smoke — implementation ready; external model dependencies required.
- **L2:** 0.5B–3B public model, fixed compute, independent held-out evaluation, multiple seeds.
- **L3:** real vLLM/SGLang + distributed RVL/verl, throughput/lag phase diagram.
- **L4:** long-horizon coding/MLSys environments with executable graders.
- **L5:** research automation where reward is independently measured downstream capability/systems improvement.

Claims must stop at the highest **measured** evidence level, not the highest implemented feature level.

## Key invariants

1. Optimization evidence and promotion evidence are separate.
2. Policy/verifier versions are first-class and stale data cannot silently train the model.
3. GRPO comparison groups are never silently truncated by replay sampling.
4. Candidate weights are transactional: train -> snapshot -> restore incumbent -> held-out compare -> promote/rollback.
5. Negative/null runs remain in the evidence ledger.
6. Protocols can be locked before results are observed.

## License

Apache-2.0
