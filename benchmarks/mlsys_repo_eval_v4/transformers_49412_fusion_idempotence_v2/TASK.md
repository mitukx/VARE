# Make module-fusion registration safe to repeat

Registering the same model fusion more than once in one process should be idempotent. A repeat registration must leave the effective patch and checkpoint-conversion mappings unchanged, while a genuinely conflicting conversion mapping must still fail clearly.

Keep the change in `src/transformers/fusion_mapping.py`. The workspace contains the pinned Transformers source and a small public regression test. Use only the allowed source file and the provided public test; do not use network access, model checkpoints, or Git history.
