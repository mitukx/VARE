# V6 Development Task Calibration — Protocol v2

Frozen cohort: `mlsys_repo_eval_v6`, protocol `vare-transformers-fresh-development-v6-protocol-v2-2026-10-21`.
Runtime: Apple Silicon host, Colima Docker `aarch64`, image `vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b`.
Each candidate ran with network disabled, 2 CPUs, 1900 MiB memory, 32 PIDs, and 120 s timeout. Grader assertions ran in the host process after the candidate container returned.

| Task | Baseline | Known fix | Baseline runtime | Fix runtime |
|---|---|---|---:|---:|
| `transformers-49315-watermark-repeat-v1` | FAIL (`repeated_ngram_count_mismatch`) | PASS | 3.60s | 3.73s |
| `transformers-49515-rope-setter-v1` | FAIL (`candidate_exception:KeyError`) | PASS | 3.56s | 3.55s |
| `transformers-49731-generation-device-type-v1` | FAIL (`structured_output_type_or_value_mismatch`) | PASS | 2.59s | 2.72s |
| `transformers-49767-janus-background-v1` | FAIL (`janus_background_color_mismatch`) | PASS | 3.68s | 3.72s |

These are fixed-patch calibration checks, not model-generated repairs. They establish grader discrimination only; they are not Coding-Agent success or held-out performance evidence.

Protocol v1 is retained as historical evidence but superseded because its RoPE worker read normalized fields after the intentional `None` assignment and rejected a valid known fix. Protocol v2 corrected the observation order and reran all four calibrations. The v1 files are not counted in the table.

Limits: all four tasks are in the Development split and originate from Transformers, so this does not establish held-out independence or repository-level diversity beyond distinct bug families. PR dates do not prove absence from model training data. The Docker image is CPU-only for these task probes; no Apple GPU task execution or speed claim is made.

Task sources: Transformers PR #49315 (watermark scoring), #49515 (RoPE compatibility), #49731 (typed output preservation), and #49767 (Janus image preprocessing). Baseline/fix SHAs, prompts, target files, and task hashes are bound by `benchmarks/mlsys_repo_eval_v6/protocol-v2.json` and its lock.
