# v12 public export smoke: retained resource aborts

The test targeted one narrow question: after v12 fetches the pinned MLX-LM commit into fresh bare Git metadata, does its public export path pass a populated detached worktree to the pinned v11 exporter and return a candidate inventory matching the frozen baseline manifest? The operation was export-only. It did not execute a candidate, call the hidden grader, load a model, or update an optimizer.

The source-boundary regressions pass, including a positive control where direct checkout runs a source repository's local smudge filter and the isolated snapshot materialization does not. The exact-head public-dispatch regression also verifies that the v11 delegate receives a populated worktree at the pinned commit. A real CLI export was attempted under two separately frozen protocols, and both attempts stopped before producing an export report:

| Protocol | Frozen RSS cap | Observed process-tree RSS | Elapsed | Swap growth | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 | 536,870,912 bytes | 545,243,136 bytes | 4.144 s | 0 MiB | `ABORTED_RESOURCE` |
| v2 | 1,073,741,824 bytes | 1,092,960,256 bytes | 7.577 s | 0 MiB | `ABORTED_RESOURCE` |

Both public export processes were terminated by the resource watchdog. The source commit fetch completed in each run (616,718 source-mirror bytes in v1; the v2 fetch log is retained). The export report, exported workspace inventory, and primary outcome were not observed. No third attempt was made after v2. The RSS monitor sums the resident sizes reported by `ps` across the process tree, so shared pages may be counted more than once; the frozen abort decisions are retained as recorded.

The frozen protocols, stdout/stderr, environment preflight, source-fetch logs, resource readings, abort classifications, source snapshots, and per-run SHA-256 manifests are under `results/mlxlm-1971-v12-public-export-smoke-v1/run-1/` and `results/mlxlm-1971-v12-public-export-smoke-v2/run-1/`. Protocol SHA-256 values are `393542f2fa8e8f19d4c652d78a2a0815c3f23ad66008846aa4ff0c07c03ca77a` and `09fd67f5959e55d893bb2a6ef1a3d34c0439b85ab043bcc4c7895936131e6920`; the run-manifest SHA-256 values are `31c03b7eadb302316448301c7221d2983be5ace117bca14b7db69fdbd8b98f05` and `27b1e55ab3c841cc70d0b0729d746c54595586f8971c28b73322f869ef2c9a3e`.

This is E0 source-level dispatch and resource evidence only. It is not grader calibration, task eligibility, model capability, or a successful export result. Reproduction commands and resource limits are frozen in each run's `protocol.json`.
