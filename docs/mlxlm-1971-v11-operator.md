# Prospective MLX-LM #1971 v11 operator interface

**Status: prospective and not executed.** This operator addresses two source
provenance defects found while reviewing v10: the historical attestation ran
Git with ambient `core.fsmonitor` configuration, and the sparse operator
checkout omitted v9 test files required by its immutable source inventory.
Protocol v10 and its lock remain unchanged and are retained as a superseded,
unexecuted protocol record.

The v11 protocol and separate SHA-256 lock are
`protocol-v11.json` and `protocol-v11.lock.json`. The contract pins the
attested operator source at `8257a4dfe2106055a048ddd4b1520a08c7e4d658` and the
historical v9 source at `3f12eefd71d1d2213b37e3a174b446cb5155e979`. It carries
forward the v10 action-specific evidence validation, immutable worktree
materialization, no-follow bounded file reads, patch snapshot, and replacement
ref protections. The sparse selection now includes `tests/`, so nested v9
attestation sees both test files listed in the frozen implementation
inventory.

The v11 CLI accepts only `grade` and `export`. For a future task calibration,
first satisfy the prospective resource and task eligibility gates, then run:

```bash
python scripts/run_mlxlm_1971_v11_operator.py \
  --action grade \
  --workspace /absolute/path/to/staged-candidate
```

For `--action export`, provide `--source-repo` and an output `--workspace`,
and optionally `--patch`. Keep grader outputs operator-side and preserve every
failed or aborted record.

The source-only regression
`test_v11_nested_v9_checkout_contains_attested_test_sources` creates a
temporary Git repository, materializes the same sparse checkout, creates the
nested v9 worktree, and verifies both required test files exist without the
skip-worktree flag. No candidate was graded to create this check.

No model was loaded, no candidate grading or calibration was performed, and
no training or optimizer update occurred for v11. Independent review and
exact-head CI remain required before any outcome-bearing use.
