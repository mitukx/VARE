# End-to-end CPU RL system benchmark v1

This benchmark executes VARE's `CapabilityLoop` through rollout, verifier, replay, candidate training, paired evaluation, and promotion or rollback. It compares that path with a deliberately narrow straight-line reference using the same seeded workload, verifier, `PrioritizedReplay`, trainer, evaluation tasks, and CPU process setup. The locked protocol is [e2e_rl_system_benchmark_v1.lock.json](../protocols/e2e_rl_system_benchmark_v1.lock.json); the runner is [benchmarks/e2e_rl_system/runner.py](../benchmarks/e2e_rl_system/runner.py), and regression coverage is in [test_e2e_rl_system_benchmark.py](../tests/test_e2e_rl_system_benchmark.py).

## What this does and does not measure

The run uses VARE's existing four-parameter contextual-bandit policy and analytic oracle. Its optimizer mutates real in-memory policy parameters, but this is a **simulated policy workload** and makes no claim about a pretrained language model, coding performance, reward-model quality, or task-success generalization. Each round gets a new held-out set of 256 generated tasks; the same set is shared by incumbent and candidate, then retired before the next round. Training and evaluation examples are disjoint by seed.

The baseline removes VARE's use-time freshness screening, fail-closed evaluation gate, candidate transaction ownership, and round serialization. It retains the same replay implementation, training hook, evaluator, and primary-gain threshold so that the comparison isolates orchestration controls more narrowly than a comparison to an unrelated system. It is not a framework baseline and does not model production serving or distributed training.

The protocol fixes twelve independent seeds, six rounds per trial, 64 rollouts per round, concurrency eight, 32 replay samples, and 256 evaluation examples per round. Arms run in isolated child processes with alternating order. The output records raw trial rows, task-sequence hashes, per-arm timing, per-round p50/p95, isolated peak RSS, Python allocation peak, policy/verifier staleness, and bootstrap intervals over seed-level throughput. A residual stage estimate is named `orchestration_replay_residual_seconds`; it includes event handling, replay bookkeeping, freshness checks, promotion evidence calculation, scheduling and other uninstrumented orchestration. It must not be interpreted as a pure CPU profile of `CapabilityLoop`.

The three-seed fault matrix injects an actual child-process exit, stale policy and verifier evidence, evaluator failure, overlapping rounds with a candidate-ID race in the reference path, and cancellation after promotion mutates the live policy but before acknowledgment. The last case is intentionally not called a rollback: without durable independent reconciliation, VARE fences the runtime, and the state remains unresolved. For evaluator and promotion fault cases, the harness installs a known positive synthetic candidate to ensure the targeted boundary is reached; those cases measure fault handling, not training.

## Reproduction

Use Python 3.11+ and the repository's normal development installation. From the repository root, choose a new output directory:

```bash
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_e2e_rl_system_benchmark.py -q
python -m benchmarks.e2e_rl_system.runner --output results/e2e-rl-system-benchmark-v1/run-1
```

The test command disables unrelated globally installed pytest plugins because one local plugin combination has an incompatible ANTLR runtime. The benchmark itself has no third-party runtime dependency. The runner refuses to overwrite an existing result directory. It writes `trials.jsonl`, `faults.jsonl`, `summary.json`, `summary.csv`, three SVG plots, and SHA-256 digests. Preserve failed runs and use a new `run-N` path for each reproduction.

For a quick smoke before the full protocol, use at least two seeds and rounds. That reduced run is a code-path check only and must not be substituted for the locked result.

## Results

The retained local run is linked below after execution. The report includes negative outcomes and limitations, including control overhead and the unresolved interrupted-promotion state. Same-host reruns establish reproducibility of the procedure, not independent reproduction by another operator or generalization to real-model training.
