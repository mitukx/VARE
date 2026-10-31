# Transformers #49458 source qualification v2

**Result: `QUALIFIED_SOURCE_LEVEL`.** The baseline failed the frozen assertion, the deliberately incomplete mutant failed the same assertion, and the upstream-fixed source passed both configuration cases.

## Protocol and environment

- Protocol: `protocols/mlsys_transformers_49458_source_qualification_v2.json` (SHA-256 `f62f163be8984323c0d237b82fbaef3cf075a80e5bb49d23fd0fd8328141ee9b`).
- Case-aware runner: `results/mlsys-fresh-development-task-qualification-v3/transformers-49458/case_runner.py` (SHA-256 `2ec495123c435fd08a8717a993170edce909f81403eb8dd45074154830c6eff7`).
- Baseline: `429e69162e2944922e45c2cea24887d32e96fbe7`; upstream fix: `927af2f73bd0d53d339f8fc0d4bd8b053f3836bb`.
- Image: `vare/fresh-qualification-transformers:0.3`, `sha256:d93a01090a15c8c9ce47d7e08c6bc935f1abb83683f6318dff866a07f08f8ff2`; Linux ARM64, Python 3.12.15, CPU Torch 2.9.1, torchvision 0.24.1.
- All arms ran once, sequentially, with network disabled, read-only source/test/runner/root mounts, 2 CPUs, 2500 MiB memory/no swap, and a 512 MiB `/tmp`.

## Arms

| Arm | `video_preprocessor_config.json` | `preprocessor_config.json` | Result |
|---|---:|---:|---|
| Baseline | Target assertion failed (47 mapping loads) | Target assertion failed (141 mapping loads) | Expected negative control |
| Wrong mutant (only class-name loop corrected) | Target assertion failed (47 mapping loads) | Target assertion failed (47 mapping loads) | Incomplete fix rejected |
| Upstream fixed | Passed | Passed | Known fix accepted |

There were no import/collection errors, timeouts, or skipped subTests in the v2 sequence. V1's aggregate pytest logs remain preserved separately; v2 records both subTest identities explicitly.

## Decision and limitations

The candidate is **source-qualified**, but remains **unassigned**. The metadata overlap review found no prior Development/Train family overlap at moderate confidence. Public pretraining exposure remains **unknown**. No model or agent inference, training, split assignment, or Held-Out access occurred. This is one CPU regression method, not a full Transformers suite result, model capability result, RL result, or novelty claim.

## Verification

From this directory: `shasum -a 256 -c SHA256SUMS`. The machine-readable summary is `qualification.json`; per-arm commands, hashes, timing, and named subTest outcomes are in the three JSON receipts.
