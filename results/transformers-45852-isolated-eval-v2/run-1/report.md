# Transformers #45852 isolated grader v2 calibration

- Protocol: `transformers-45852-isolated-eval-v2`; SHA-256 `9029a4ff77063551f91dbc8273f5818ad3ccd7a3375be7788ce9b529deb8a5d1`.
- VARE implementation commit: `05ac1ec2136701909abf5e5e7b2b6c793f4f0964`; working tree clean at run start: `True`.
- Upstream baseline/fixed commits: `381032b7e0a8896ae48f90dbe83eae3bf7630953` / `909ece888ed4b55402dc8acbb9c13ddf8ff977b5`.
- Colima Docker server: 29.5.2, linux/aarch64; exact runtime image: `localhost:5000/vare/transformers-cpu@sha256:5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f`.
- Total wall time: 216.60s. No LLM inference or training was run.

## Calibration

- Baseline: **FAIL**, as required. Ordinary 2D padding and NumPy-square are controls that pass; Torch square, asymmetric Torch, nonuniform Torch and nonuniform NumPy cases fail on the pinned baseline.
- Exact upstream known fix: **PASS**, 6/6 grader cases. This is calibration, not an agent-generated repair.
- Candidate behavior test: **PASS**; all six constant/mean mask and corrupted ordinary-padding mutations were rejected.
- Isolation red-team: **PASS**, all 7 checks passed (no grader/Git/secret/network access, output forgery, timeout, memory cap, path traversal, hidden-test injection, symlink).
- Peak memory was not measured by the current harness.

## Reproduction

From a clean checkout of the recorded VARE commit with the exact Colima image and local pinned source mirror:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/run_transformers_45852_calibration.py \
  --source-repo "$HOME/.cache/vare/transformers-debug.git" \
  --output results/transformers-45852-isolated-eval-v2/run-1 \
  --image 'localhost:5000/vare/transformers-cpu@sha256:5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f'
```

The command creates a new output directory and does not overwrite previous runs. Raw per-case and red-team records are in `calibration.json`; source and implementation snapshots are under `snapshots/`. `evidence-manifest.json` hashes every retained file in this run except itself.

## Limits

This validates only the pinned Transformers behavior and the declared isolated Docker contract. It does not establish arbitrary Python sandbox soundness, full upstream test-suite equivalence, model-generated success, or policy improvement.
