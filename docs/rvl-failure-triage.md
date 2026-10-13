# Read-only RVL replay diagnostics: independent audit binding

This bridge extends the standalone [failure triage](failure-triage-v1.md) to RVL's existing read-only TokenReplay reader. It is a diagnostic contract, **not** an automatic learning loop.

## Workflow

1. Obtain a ready RVL SQLite replay database and specify externally observed current policy/verifier versions.
2. Generate a manifest:
```bash
vare rvl-triage --replay-sqlite /path/to/replay.sqlite --current-policy-version 6 --current-verifier-version 5 --output /tmp/replay-manifest.json
```
3. Separately adjudicate each generation. Create `/tmp/audit.json` containing exactly `selection_sha256` (from manifest) and `rows`, one row per `rows[].sample_id` with `sample_id`, the corresponding `response_sha256`, and **independently determined** Boolean `format_valid` and `trusted_pass`. Do not copy proxy decisions as labels.
4. Provide `/tmp/bindings.json` containing `run_id`, `training_cohort_sha256`, `heldout_cohort_sha256`, `proxy_grader_sha256`, `trusted_grader_sha256`, `active_tokens`, and `clipped_tokens`. These are externally provided values and are not attested by the bridge.
5. Analyze:
```bash
vare rvl-triage --replay-sqlite /path/to/replay.sqlite --current-policy-version 6 --current-verifier-version 5 --audit /tmp/audit.json --bindings /tmp/bindings.json --output /tmp/diagnosis.json
python -m pytest tests/test_rvl_failure_triage.py
```

## Hard boundaries

- Identity is the encoded pair `[RVL group ID, generation index]`. Repeated prompt IDs therefore remain distinct.
- The manifest hashes the selected rows and explicit current versions. Changing a response or replay selection invalidates previously prepared labels. This does not attest to a concurrent writer's source state or a database snapshot across separate commands.
- The complete ready selection must be adjudicated; missing/extra/duplicate identities, hashes, malformed types, and identical training/evaluation or proxy/trusted grader hashes fail closed.
- Storage groups are **not assumed GRPO comparison groups**. Only multiple same-prompt, same-task, same-policy, same-verifier members are eligible for reward-contrast diagnostics; even those need independent confirmation of true training group membership.
- The source RVL replay is read-only. No model training, verifier modification, policy promotion, or rollback is performed.
- The supplied training/held-out hashes do not establish sample disjointness; independent gold-label quality is an external requirement. Audit results cannot be described as held-out model improvement or causal intervention effectiveness.

## Next research gate

Use a new externally adjudicated workload and frozen thresholds. First measure precision/recall of warning signals on controlled failures, then compare costly intervention proposals to no-op/simple baselines at matched budgets. Only after that should a preregistered real-model experiment be considered.
