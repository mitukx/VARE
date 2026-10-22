# MLX-LM #1912 prospective v4: macOS/Colima calibration runbook

**Status:** executable preparation only; **no live v4 calibration has been performed by this change**. PR [#98](https://github.com/mitukx/VARE/pull/98) must not be merged solely on static hashes or GitHub CI.

## Hard prerequisites

- Apple Silicon macOS with Colima's Docker context, pinned Docker image already built/present, and the externally pinned `ml-explore/mlx-lm` Git checkout containing both pinned revision objects.
- Use this **PR #98 checkout**, not `main` (which may not contain prospective v4). Ensure a clean repository working tree; frozen input/implementation hashes are checked by `verify_protocol_v4`.
- Do **not** override `TMPDIR` to conceal Colima host-path mount defects. Verify Docker network isolation, candidate UID/read-only file mounts, and no hidden oracle availability in the retained red-team report.
- Create only the parent directory of a **new** output path; never reuse or overwrite an earlier run. Keep frozen v1–v3 evidence untouched.

## Run on the Apple Silicon host

```bash
# Set this to the already-available pinned upstream checkout on your machine.
SOURCE_REPO=/absolute/path/to/pinned/mlx-lm
mkdir -p results/mlxlm-1912-isolated-v4
OUTPUT=results/mlxlm-1912-isolated-v4/run-1

# Read-only prereq checks (fails closed on missing pinned commits or hash drift):
PYTHONPATH=src python scripts/preflight_mlxlm_1912_v4.py \
  --source-repo "$SOURCE_REPO" --output "$OUTPUT" --docker-context colima

# New calibration + intentionally wrong candidate in the isolated grader:
PYTHONPATH=src python scripts/preflight_mlxlm_1912_v4.py \
  --source-repo "$SOURCE_REPO" --output "$OUTPUT" --docker-context colima --execute
```

The helper compares baseline/fixed Git blob bytes to the independently pinned SHA-256 with Git replacement objects disabled, checks the v4 protocol and locked evaluator implementation hashes, validates the exact Docker image and refuses existing evidence directories. It runs the existing calibration CLI (baseline/known fix/red team) then constructs a **new negative-control candidate** returning index-zero at value zero for all inputs, and grades that candidate with v4.

On success, `$OUTPUT/calibration.json` must report `calibration_passed=true` with **baseline false, exact known-fix true, source swap rejected and red team passed**. The separate `$OUTPUT/adversarial-negative-control.json` must report `adversarial_rejected=true` and include `shifted_maximum_nonzero_values` in the failed-case list. `adversarial-negative-control.sha256` hashes that report. Verify this evidence is bound to the final PR HEAD, image digest and source SHA(s) before the PR review thread is resolved. A failed run stays retained with its true status, and a new attempt uses `run-2`, etc.

**Limitations:** the helper is prospective operational code and is not itself a frozen oracle or a model experiment. A successful negative control establishes only this narrow anti-shortcut property. It does not establish full top-p correctness, hosted production isolation, Metal equivalence, or policy improvement. Tests / CI and frozen v4 correctness checks still apply. The helper does not run model inference or RL optimization.
