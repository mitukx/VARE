# Prospective MLX-LM #1971 v12 operator interface

**Status: prospective and not executed.** A source-level red-team check found that the v11 export path passed a caller-supplied Git repository to the historical exporter. Its `git worktree add` could execute a smudge filter selected by that repository's `.git/info/attributes` and `.git/config` before source verification.

The v12 wrapper fixes this boundary before delegating to v11: it fetches only the protocol-pinned baseline commit into fresh mode-0700 bare Git metadata, disables replacement refs and upload-pack hooks, and never checks out files using the input repository's metadata. The resulting baseline checkout uses the fresh repository's empty local-attribute and filter configuration. The source-only regression has a positive control that observes the input repository's smudge filter during direct checkout, then verifies that the v12 snapshot and checkout do not run it.

A separate source-provenance preflight fetched the pinned MLX-LM baseline, passed it through the v12 snapshot helper, and verified all 201 Python files against the existing frozen source manifest. The retained record is in `results/mlxlm-1971-v12-real-source-snapshot-v1/`. It verifies source identity only and does not invoke the candidate grader. Temporary object/worktree storage was 3.81 MB and the run took 3.92 seconds; Git transport bytes were not measured, as recorded in the append-only resource erratum.

`protocol-v12.json` and its lock retain the exact baseline commit, v11 operator and protocol hashes, allowed actions, and claim limits. The prospective v11 protocol and lock remain unchanged.

For a future export, invoke:

```bash
python scripts/run_mlxlm_1971_v12_operator.py \
  --action export \
  --source-repo /absolute/path/to/upstream-clone \
  --workspace /absolute/path/to/new-candidate-workspace
```

For grading, invoke the same command with `--action grade --workspace ...`. Keep candidate output and grader evidence operator-owned. The checkout running this bootstrap must itself be trusted.

No candidate was graded, no model was loaded, and no calibration, inference, training, or optimizer update was performed for v12. This source-boundary regression does not certify the historical v9 grader or make the consumed #1971 task eligible for training or held-out evaluation.
