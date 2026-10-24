# Crash-harness test portability correction v3

The first exact-head CI run failed during collection on Python 3.11, 3.12, and 3.13 because the crash-evaluator regression imported optional PyTorch at module scope. The local M1 environment has PyTorch, but the repository's generic pytest jobs do not install it.

The evaluator fixture now calls `pytest.importorskip("torch")` inside the PyTorch-dependent test. The replay-priority ordering regression remains dependency-free and runs in the base test environment. Local focused tests pass. No experiment runner, protocol, measured output, or numerical acceptance rule changed; the original measured crash bundle remains bound to correction v2. `crash-runner-correction-v3.json` binds the current test and runner hashes for CI and future reproduction.
