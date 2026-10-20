# Separated coding eval, Mac measurement

This directory is a real local-model measurement, and the outcome is a negative baseline. It is not a training result and not held-out improvement.

| Item | Value |
| --- | --- |
| Host | Mac M1, Colima, Python 3.12, CPU |
| Code | PR #85 head `b15f52ac2f24948238e3d9c964f6ec99d03b9e26` |
| Pack | `c8c0488980425e6d316abc21dfc136f16c7e265b` |
| Image | `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258` |
| Model | Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`, seed 101, greedy |

E0 harness matched. The canary was visible in 0 of 3 probes, each with Docker return code 0 and a single readonly `/workspace` mount (`harness.json`). Unpatched train tasks passed 0 of 5 (`unpatched.json`).

Primary train tasks passed 0 of 5. `wilson_interval(0, 5)` is `[0.0, 0.43448246478317476]`, which rounds to `[0, 0.43]`. All five diagnoses are `no_source_change` with `stop_reason` `no_tool_call_retries_exhausted`, 6 tool turns, and agent wall times of 31.32 s, 28.59 s, 27.41 s, 31.00 s, and 30.94 s (`primary-tasks.json`).

Qwen2.5-Coder-0.5B did not run. After the primary rows were written, `mem_available_bytes()` raised `FileNotFoundError` because `/proc/meminfo` is absent on macOS (`vare-pr85-eval.log`). The median task wall was under the 90-second gate. The memory comparison itself did not run, so this directory does not say whether the 6 GiB condition would have passed.

Workspace, snapshot, canary, and staged hidden-evaluator trees from the tarball are not included. Scores and agent records are the JSON files. The pack commit remains the source of the hidden tests.
