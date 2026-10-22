# Transformers #45852 isolated grader v3 calibration

- Protocol ID: `transformers-45852-isolated-eval-v3`
- Protocol SHA-256: `512575c223a5bd3edebe762f6d48e97daaceda9acf3c2713e3f1b81644c25779`
- VARE implementation commit: `b9f5dd6b32bacad026b87a5fed43dac2108e142a` (clean tree at start: `True`)
- Upstream baseline/fix: `381032b7e0a8896ae48f90dbe83eae3bf7630953` / `909ece888ed4b55402dc8acbb9c13ddf8ff977b5`
- Runtime: Colima Docker 29.5.2, Linux/aarch64, exact image `localhost:5000/vare/transformers-cpu@sha256:5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f`
- Calibration wall time: 216.83 seconds. Model inference/training: not run.

## Result

- Pinned baseline: **FAIL** overall; it fails Torch square, asymmetric Torch, and nonuniform Torch cases. Ordinary 2D padded batching and NumPy cases remain working controls.
- Exact known fix: **PASS** (6/6 parent-owned cases). The known-fix replay calibrates the grader and is not an agent-generated repair.
- Shortcut/corruption mutants: **PASS**; rejected 6/6: `constant-mask-output, constant-numpy-mask-output, corrupt-padded-attention-mask, corrupt-padded-input-ids, mean-mask-output, mean-numpy-mask-output`.
- Isolation red-team: **PASS**, 7/7 controls. Grader/Git/secrets/network stayed inaccessible; forged success, timeout, memory exhaustion, traversal, injected tests and symlinks were rejected.
- Pinned source SHA mismatch unit regression: included in the v3 implementation lock and passed in the focused suite. Protocol bytes and implementation manifest mismatch regressions also passed.
- Peak memory was not directly measured; the Docker memory ceiling is enforced at 1536 MiB.

## Reproduction

Run on the recorded commit with the exact image and pinned source mirror:

```bash
DOCKER_CONTEXT=colima PYTHONPATH=src python scripts/run_transformers_45852_calibration.py \
  --source-repo "$HOME/.cache/vare/transformers-debug.git" \
  --output results/transformers-45852-isolated-eval-v3/run-1 \
  --image 'localhost:5000/vare/transformers-cpu@sha256:5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f'
```

`calibration.json` contains raw per-case outputs, wall times and red-team results. The `snapshots/` directory contains every locked implementation file, protocol and both exact upstream target sources. The evidence manifest hashes all run files other than itself. The command refuses an existing output path.

## Scope limits

This is a narrow historical-source calibration. It does not establish arbitrary Python sandbox soundness, full Transformers test-suite equivalence, model-generated success, or policy improvement.
