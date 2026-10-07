# Experimental program

## North-star metric

`held_out_capability_gain / GPU_hour`

Every headline result also reports wall-clock time, rollout tokens, learner tokens, verifier cost, failure rate, lag distributions, verifier versions, and promotion/rejection counts.

## Locked L2 protocol

The first real-model campaign is frozen in `protocols/l2_rvl_qwen_v1.lock.json`. The SHA-256 is computed from hypotheses, configuration, primary metrics and acceptance rules before results are observed. Any later change creates a new protocol version rather than overwriting the lock.

Required fixed-compute arms:

1. uniform curriculum + fixed verifier;
2. failure-driven curriculum + fixed verifier;
3. failure-driven curriculum + joint policy/verifier freshness.

Run all declared seeds and retain negative/null runs.

## Required ablations after L2

1. no freshness control vs policy-only freshness vs joint policy+verifier freshness;
2. single verifier vs verifier ensemble;
3. FIFO replay vs VARE priority replay;
4. group-breaking replay vs group-preserving replay as a correctness sanity check, not an optimization claim;
5. always-promote vs trusted fail-closed promotion gate;
6. fixed verifier vs co-evolving verifier refresh.

## Evidence ladder

- **L0:** deterministic CPU demo (completed).
- **L1:** tiny causal-LM smoke test; no capability claim.
- **L2:** public 0.5B–3B model, held-out tasks, multiple seeds.
- **L3:** real vLLM/SGLang rollout + RVL/verl trainer, GPU throughput and lag phase diagram.
- **L4:** long-horizon coding/MLSys environment with independent executable graders.
- **L5:** research-automation task where reward is an independently measured downstream capability or systems improvement.

A feature is not evidence. Headline claims are allowed only when the corresponding raw artifacts and immutable protocol are retained.
