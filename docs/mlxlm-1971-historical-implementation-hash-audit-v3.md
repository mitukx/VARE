# MLX-LM #1971 historical implementation hash audit v3

Version 3 repeats the v4–v9 audit after applying `GIT_NO_REPLACE_OBJECTS=1` to every Git subprocess, including the audit's own `HEAD` identity lookup. The append-only report records source commit `e0dec877b4fb2da53a5766744bc06873d2a7a3ba` and verifies 85 implementation hashes across six frozen protocol versions.

Versions 1 and 2 and all frozen task protocols and calibration evidence are preserved. The v3 report and checksum manifest are in [`historical implementation audit v3`](../results/mlxlm-1971-historical-implementation-audit-v3/).
