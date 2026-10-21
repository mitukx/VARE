# Isolated Public-Test Calibration

The controller now imports only `/workspace/src` and executes zero-argument functions whose names start with `test_` from the immutable `test_public.py`. The public file remains mounted read-only, the source workspace is mounted read-only, and the container remains network-disabled. This runner behavior is separate from hidden grading.

| Task | Baseline | Known fix | Baseline seconds | Fix seconds |
|---|---|---|---:|---:|
| `transformers-49315-watermark-repeat-v1` | FAIL (exit 1) | PASS | 4.25s | 4.35s |
| `transformers-49515-rope-setter-v1` | FAIL (exit 1) | PASS | 4.24s | 4.27s |
| `transformers-49731-generation-device-type-v1` | FAIL (exit 1) | PASS | 3.10s | 3.13s |
| `transformers-49767-janus-background-v1` | FAIL (exit 1) | PASS | 3.94s | 3.93s |
