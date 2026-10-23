# MLX-LM #1971 historical implementation hash audit v2

Version 2 repeats the v4–v9 audit with `GIT_NO_REPLACE_OBJECTS=1` applied to every Git object read. Its regression creates a replacement ref from a frozen commit to a coherently modified commit and proves the audit still reads the original object and rejects the replacement worktree bytes. The retained report verifies all 85 implementation hashes across six versions and is bound to source commit `1b346b7be04dad87e7726f888963d006e9063f7f`.

Version 1 and all frozen task protocols and calibration evidence are preserved. The append-only v2 report and SHA manifest are in [`historical implementation audit v2`](../results/mlxlm-1971-historical-implementation-audit-v2/).
