# Protocol v10 resource-limit correction

Protocol v9 declared `cpus=1` and `pids=32`, but its isolated candidate call did not forward them; `IsolationLimits` therefore used the default PID ceiling of 64. Protocol v9 remains immutable and is superseded for evidence requiring the declared resource contract. Protocol v10 preserves the same task definitions, image, and grader semantics while forwarding CPU and PID limits to the isolated process and recording them in every result.

## Calibration

| Task | Baseline | Known fix | Container limits |
|---|---|---|---|
| BLOOM static cache | fail (`candidate_exception:ValueError`) | pass | 1 CPU, 1900 MiB, 32 PIDs, 120 s |
| Phi3 LongRoPE | fail (`cached_full_prefix_logits_mismatch`) | pass | 1 CPU, 1900 MiB, 32 PIDs, 120 s |

The exact frozen ARM64 image digest and resolved image ID are in each raw JSON result. The v10 hostile-import red-team driver passed all five cases, including candidate-originated forged stdout, sitecustomize, changed imported module, symlink, and forged result file.

## Limits

This corrects task grader execution limits; it does not establish host-kernel isolation or general arbitrary-repository sandbox safety. The two task adapters preserve their existing narrow semantics. No policy update or capability improvement is claimed.

## Reproduction

Prepare the known-fix workspace from the two frozen upstream commits before running the red-team suite. The workspace preparer extracts only the pinned source tree and public fixture; the red-team script records a deterministic SHA-256 digest of every file, directory, and mode in that input.

```bash
git clone --filter=blob:none https://github.com/huggingface/transformers.git /tmp/vare-transformers-source
git -C /tmp/vare-transformers-source fetch origin 536ecc007387a50e77603bb5d92100e9b07514cc bc4d8df2c4604727ea63414c7af48de7e6eb9c0b
PYTHONPATH=.:src python scripts/prepare_transformers_dev_cohort_v1.py --task-id transformers-49481-bloom-static-cache-v1 --kind known-fix --repository /tmp/vare-transformers-source --output /tmp/vare-bloom-known-fix
PYTHONPATH=.:src DOCKER_CONTEXT=colima python scripts/grade_transformers_fresh_dev_v5.py --task-id transformers-49481-bloom-static-cache-v1 --workspace /tmp/vare-bloom-known-fix --image 'vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b' --output results/phase-a-integrity-v10/result.json
PYTHONPATH=.:src DOCKER_CONTEXT=colima python scripts/run_transformers_fresh_dev_v10_redteam.py --workspace /tmp/vare-bloom-known-fix --output results/phase-a-integrity-v10/redteam-rerun.json
```
