# VARE-IG research package

This package records the VARE-IG reward-informativeness pilot, real Qwen3-4B MLX training, durable update recovery, and the paired long-horizon Development cohort.

- [Research summary](summary.json): machine-readable per-task, per-seed, scheduler, recovery, and long-horizon outcomes.
- [Evidence files](evidence/): saved completions, verifier results, behavior-sample records, scheduler selections, checkpoint JSON manifests, recovery decisions, and event-ledger exports. Large tables and full traces are gzip-compressed; source and uncompressed content hashes are recorded.
- [Evidence verifier](../../scripts/verify_vare_ig_evidence.py): checks published hashes and both hash-chained event exports.
- [Run source snapshot](run_source_snapshot/): exact 14-file source set named by the real training locks.
- [Protocol redaction manifest](evidence/protocol-redaction-manifest.json): original and portable-copy hashes for machine-specific protocol paths.
- [Local-only artifacts](evidence/local-only-artifacts.json): hashes/sizes for adapter weights, optimizer weights, and the original trajectory databases. These large binary artifacts are kept in the local research results directory.
- [Full report](../../docs/vare-ig-flagship-research-2026-11-02.md): methodology, results, limitations, and reproduction commands.

All task results are synthetic coding contracts. The report does not claim open-source repository semantics, held-out capability gains, a novel RL algorithm, or a general inference speedup.
