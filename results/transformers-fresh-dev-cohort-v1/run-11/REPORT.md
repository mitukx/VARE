# Fresh Development task calibration — protocol v9

Status: task/grader qualification only. No agent rollout, model inference, SFT, or GRPO was run under this protocol.

## Frozen conditions

- Protocol: `vare-transformers-fresh-development-cohort-v9`
- Protocol SHA-256: `5a143f945afca9fe15c0eb0af250e0c8010331fd21500baee720f47c7faf807b`
- VARE source at start: `e3adb197e05d566536f826c343970d1c35dbb47e`
- Runtime image: `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Local daemon image ID: `sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`
- Runtime: Colima Docker, arm64 Linux, one CPU, 1,900 MiB memory limit, network disabled, read-only root, 120 second task timeout.

## Calibration outcomes

| Task | Baseline independent grade | Known fix independent grade | Baseline public test | Known fix public test | Grader wall time |
| --- | --- | --- | --- | --- | ---: |
| `transformers-49481-bloom-static-cache-v1` | fail (`ValueError` during static-cache generation) | pass | fail | pass | 4.00s / 4.05s |
| `transformers-49361-phi3-longrope-v1` | fail (cached and full-prefix logits differ) | pass | fail | pass | 4.10s / 4.03s |

The host-side checks preserve the frozen task criteria: Bloom generation returns three new tokens and retains the input prefix; Phi3 cached logits match full-prefix logits at four decode steps for both the unset and five-token sliding-window settings (`rtol=atol=1e-4`). Candidate code returns observations through VARE's nonce-framed isolated-candidate worker. The worker contains no assertions or expected outputs; comparisons remain in the host process. The public test file, task adapter, evaluator, protocol, expected results, and `/grader` are not staged into the candidate subject.

## Local model preflight

- Host: Apple Silicon `arm64`, 32 GiB unified memory; system memory pressure reported 50% free before any model load.
- MLX device selection: `Device(gpu, 0)`; no CPU fallback was selected.
- Environment: isolated Python 3.12.12 venv, `mlx-lm==0.32.2`, `mlx==0.32.4`; full dependency freeze is retained alongside this report.
- Checkpoint: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, cached snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; `model.safetensors` is 2,263,022,417 bytes, SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`.
- The Metal runtime was imported and device enumeration succeeded. The checkpoint was not loaded, no token was generated, and no model inference or training was performed.
- A separate Development-only coding-agent preregistration is frozen as protocol v3 (`benchmarks/transformers_coding_agent_feasibility_v3/protocol-v3.json`, SHA-256 `14fd65da5f023da0bc488733a83a209fe2cf04523331f51fedcbafc2401fe40a`). Earlier v1/v2 preregistrations were superseded before inference because controller review identified model-feedback and diff-budget defects. The current v3 binds controller source commit `1d354331ce25a3ecabb564b078c8306538aa86b8`; inference remains gated on current-head CI and independent review.
- `SHA256SUMS-v4` is the current complete evidence manifest, including the no-overwrite task-preparation guard and regression test. Prior manifests are retained as historical snapshots.

## Adversarial checks

`run-12/redteam.json` records five checks. A hostile target-source import probed for grader/test/adapter/Git files, secret-like environment variable names, network, writable subject paths, and forged stdout. It was graded using the actual isolated runner and passed only because all isolation assertions held and the host still computed the grade. Separate candidate trees containing `sitecustomize.py`, a modified imported `cache_utils.py`, a symlink, or a forged grade file were rejected before candidate execution. All 5 checks passed.

## Superseded attempts and limitation

- Protocol v6 staged its task adapter beside candidate code. Its task calibration is retained as historical functional evidence but does not establish grader secrecy. Do not use it for a trust-boundary claim.
- Protocol v7 failed after candidate execution because its result serializer could not encode `CandidateException`.
- Protocol v8 failed after candidate execution because the report referenced the v2 implementation-lock key.
- The initial invocation also omitted `PYTHONPATH=.` and failed before grading. These attempts are retained in `run-9`, `run-10`, and `run-11/superseded-attempts.json` rather than removed.
- The task cohort contains two tasks from one repository, so cross-repository independence and broad coding generalization are unproven. The task patches are public upstream history; model pretraining exposure is unknown.
- The Bloom grader intentionally retains the original weak output-shape/prefix criterion. It proves the original calibration behavior, not broad behavioral equivalence or robustness to a deliberately hard-coded output implementation.
- These checks qualify the boundary for the documented candidate interface; they do not constitute a general-purpose sandbox certification.

## Reproduction

From the VARE repository root, with Colima and the locked image present:

```bash
PYTHONPATH=. DOCKER_CONTEXT=colima python scripts/grade_transformers_fresh_dev_v4.py \
  --task-id transformers-49481-bloom-static-cache-v1 \
  --workspace /path/to/workspace-baseline-v6 \
  --image 'vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b' \
  --output /path/outside/workspace/bloom-baseline.json
```

Use the corresponding task ID and workspace for Phi3 and the known-fix workspaces. The v9 red-team command is `PYTHONPATH=. python scripts/run_transformers_fresh_dev_v9_redteam.py`.
