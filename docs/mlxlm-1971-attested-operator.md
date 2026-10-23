# Attested historical MLX-LM #1971 v9 calibration operator

**Status: prospective operator path only; no new calibration is claimed by this document.**

The v9 protocol, lock, historical source and historical calibration evidence are immutable. Changing a v9-hashed implementation file in place would invalidate its own source-lock checks. Do not "repair" this by resealing v9.

The new `scripts/run_attested_mlxlm_1971_v9_calibration.py` creates a private temporary Git worktree at the fixed, full v9 source commit `3f12eefd71d1d2213b37e3a174b446cb5155e979`. It disables replacement refs in every Git operation and in the launched legacy process, verifies all v9 protocol/lock/evaluator bytes against independent Git objects, then runs the original v9 calibration **from that pinned worktree**. The output path must not already exist; legacy evidence under earlier run directories is untouched.

Example (on a configured machine with the pinned upstream checkout and Colima image available):

```bash
python scripts/run_attested_mlxlm_1971_v9_calibration.py \
  --source-repo /absolute/path/to/mlx-lm \
  --output /absolute/path/to/new-append-only-calibration
```

The same operator module now exposes `run_pinned_v9_task("grade", workspace)`
and `run_pinned_v9_task("export", workspace, source_repo=source_root, patch_path=patch)`.
Both run the historical public task API **inside the separately launched fixed-commit
Python process**, after the Git-object attestation. The outer implementation does
not import the mutable current-checkout public grader to execute either operation.
The subprocess uses Python isolated mode (`-I`), explicitly imports the
pinned `src/vare` code, requires successful exit and a single JSON object
report, and rejects extra/forged output. It keeps operator-side evaluation
results under the caller's control; callers must never send hidden results or
grader artifacts back into a candidate workspace. This is still a narrowly
scoped operator API, not a new frozen v10 study.

The wrapper passes through the legacy process return code and fails closed on missing source, conflicting output, source-anchor mismatch, Git worktree failure, and historical attestation failure. Unit tests do not load a model, invoke Docker, create real calibration output, or prove that the full environment can reproduce prior grader outcomes.

**Unresolved:** this does not replace the direct public `vare.mlxlm_1971_repo_task.grade_workspace` or `export_candidate` API and does not create a separately frozen v10 grader protocol. Those legacy entrypoints must not be relied on for a new provenance-sensitive study. A prospective v10 API and independent review remain required; this operator wrapper is a narrowly scoped mitigation for the calibration path.
