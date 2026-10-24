# MLX-LM #1971 historical implementation hash audit v1

The auditor verifies the frozen protocol and lock bytes for v4–v9 against their recoverable Git source commits, then compares every implementation hash declared by each protocol against the corresponding Git blob at that commit. The retained audit verified 85 implementation-file hashes across six versions. No protocol, lock, calibration, or historical result was rewritten.

The regression suite also constructs a committed protocol/lock/implementation and proves that coherently changing the implementation hash in both working-tree JSON files is rejected because the bytes no longer match the pinned Git objects.

Reproduce with:

```bash
python scripts/audit_mlxlm_1971_historical_implementation_hashes.py \
  --output /tmp/mlxlm-1971-historical-audit.json
```

The append-only repository result is [`audit.json`](../results/mlxlm-1971-historical-implementation-audit-v1/audit.json), bound to commit `bc457444869ca63a316ec0af1270d81c8e00fe25`; its adjacent `SHA256SUMS` verifies the report.
