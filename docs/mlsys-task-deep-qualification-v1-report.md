# MLSys/Coding Task Deep Qualification v1

## Decision

**NO-GO for SFT/GRPO and model rollouts.** Deep review expanded the 24-item provisional pool with three newer MLX-LM changes and reproduced 15 distinct source-level regressions at pinned baseline and fixed revisions. The source-task allocation is frozen as 3 train, 3 development, and 9 held-out tasks in [`protocol.json`](../results/mlsys-task-deep-qualification-v1/protocol.json). The qualification run itself launched no model, but a later retrospective audit established prior model exposure on all three Development tasks; see the correction below. The nine held-out tasks remain unused by model or agent.

This is enough to retain a 15-task source pool, but it is not an executable RL benchmark yet. VARE's current isolated candidate route supports a narrow set of JSON-compatible function contracts. It cannot safely accept repository patches while keeping arbitrary pytest graders separate from the candidate workspace. The upstream test files are pinned and hashed here, but have not been installed as hidden graders behind VARE's boundary. Do not start task rollouts, SFT, or GRPO until that narrow grading path is implemented and audited.

## Frozen qualification evidence

The machine-readable catalog records each upstream PR, baseline and fixed commit, grader test path and selector, SHA-256 for the grader and repository license, split, reproduction outcome, and raw-log paths in [`qualification.json`](../results/mlsys-task-deep-qualification-v1/qualification.json). Raw base/fixed outputs are retained under [`raw/`](../results/mlsys-task-deep-qualification-v1/raw/). The original survey v1 pool and protocol were not edited.

Verify the retained files with `cd results/mlsys-task-deep-qualification-v1 && shasum -a 256 -c SHA256SUMS`. `evidence-manifest.json` also records the exact path inventory, including the SHA256SUMS file.

| Source task | Base → fixed | Paired process wall time (s) | Grader focus |
| --- | --- | ---: | --- |
| Transformers #45852 | fail → pass | 5.20 → 4.96 | 4D attention-mask collation |
| Transformers #48319 | fail → pass | 5.83 → 5.92 | stale best-checkpoint recovery |
| Transformers #49208 | fail → pass | 6.44 → 6.44 | metrics after checkpoint resume |
| Transformers #49412 | fail → pass | 4.84 → 5.24 | fusion registration idempotence |
| MLX #4662 | fail → pass | 0.32 → 0.22 | convolution input-gradient shape |
| MLX-LM #1795 | fail → pass | 1.56 → 1.53 | MoE routing-index gradients |
| MLX-LM #1902 | fail → pass | 1.71 → 1.66 | continuation log-likelihood after truncation |
| MLX-LM #1904 | fail → pass | 3.00 → 3.09 | dict-valued tool-call arguments |
| MLX-LM #1912 | fail → pass | 1.50 → 1.46 | retain top token at tiny top-p |
| MLX-LM #1921 | fail → pass | 1.56 → 1.50 | padded-batch log-probabilities |
| MLX-LM #1926 | fail → pass | 1.58 → 1.49 | Falcon-H1 cache initialization and gradients |
| MLX-LM #1935 | fail → pass | 3.86 → 2.98 | zero `max_tokens` API behavior |
| MLX-LM #1938 | fail → pass | 1.78 → 1.58 | Phi3Small vocabulary-mask axis |
| MLX-LM #1971 | fail → pass | 1.60 → 1.48 | MiMo attention value scaling |
| MLX-LM #1979 | fail → pass | 1.53 → 1.46 | SSM right-padding state equivalence |

Times are wall-clock for starting Python/pytest and running the selected tests; they exclude environment construction and model rollout. Transformers tests used CPU PyTorch with tiny synthetic fixtures. MLX-LM tests ran on Apple Silicon macOS with MLX 0.32.4 / mlx-lm 0.32.0 and `mx.default_device() == gpu`; the MLX core convolution test used a CPU-only build because the installed Xcode Command Line Tools do not provide Apple's `metal` compiler. No pretrained checkpoint or external service was used by the selected regression tests. These timings are not inference throughput, optimizer-step cost, or GPU-hour measurements.

The selected Python package versions are retained in the three environment lock snapshots alongside the catalog. The Transformers source revisions require different package versions; each replay prepended that exact checkout's `src/` directory to `PYTHONPATH`. The replay command, selected test path, raw output path, and raw-log hash are recorded per task.

Repository licenses at the pinned revisions are Apache-2.0 for Transformers and MIT for MLX and MLX-LM. The selected tests use synthetic/local fixtures and add no model weights or external datasets. Candidate-level task prompts and executable environment locks are not yet frozen.

## Deduplication and exclusions

The 15-task pool counts one representative per overlapping family: #1904 subsumes #1910's tool-argument parser case; #1795 subsumes #1930's routing-index-gradient case; and #1921 is retained while #1925's related padding-mask task is excluded. The newly probed #1959 ERNIE correction-bias test detects its change, but is not counted separately because it is another MoE routing task. #1939's PhiMoE router case is likewise excluded; its merge range also contains the Phi3Small mask change counted as #1938.

Other exclusions were evidence-based:

- #1787's added test passes on the pre-fix source and therefore does not detect its claimed regression.
- MLX #4668's memory-safety test passes on the pre-fix source without instrumentation.
- Transformers #49116 requires remote 9B–2.4T model checkpoints; #49147's target test is skipped on both revisions.
- TRL #7768's target LoRA/MoE test is skipped on both revisions; Transformers #49500 requires distributed multi-GPU sharded saves.
- vLLM #52626 depends on CUDA/vendor kernels unavailable on M1; #56841's Linux connector stack is not represented by this local Apple runtime.

The pool contains no PR #84–#89 task reuse. All retained changes postdate the Qwen3 4B model-training completion proxy of 2025-08-09. That date is only a freshness proxy, not a published exact training-data cutoff. The source PRs and tests are public; the future agent/evaluator runtime must run offline and keep grader files out of the candidate workspace to prevent retrieval or direct test leakage.

## Split and next gate

The frozen source split is Train 3 / Development 3 / Held-out 9. The exact membership is in `protocol.json`. Held-out tasks remain unused by any model or agent; only the human qualification step replayed their pinned regression tests against baseline and fixed source revisions. The held-out task count is too small for an unqualified statistical-superiority claim; any eventual report must provide the absolute rate difference and uncertainty.

Before any Qwen3-4B MLX development-only pass@k run, implement and audit a repo-task path that (1) stages candidate patches in a resource-limited, network-disabled container, (2) mounts only the baseline source and candidate workspace there, (3) runs hash-verified grader files from a separate read-only location, (4) returns structured outcomes without exposing grader contents, and (5) records time, memory, failures, and exact source/dependency hashes. Then use development tasks only to measure pass@k, reward variation, inference latency/tokens, and a small real optimizer-update cost pilot. Make the GO/NO-GO decision before any held-out access or larger SFT/GRPO run.

Until that gate passes, no-update baseline, SFT, GRPO, agent rollouts, and held-out evaluation remain unstarted. This NO-GO is reproducible from the pinned source replays and the declared missing safe grading boundary; it is not a claim that policy updates cannot improve ability.

## Retrospective exposure correction — 2026-10-31

The earlier statement that no model or agent had seen any task in this pool was incorrect for the three tasks allocated to Development. A retrospective identity match against the retained Stage 2/3 exposure ledger found prior model attempts on all three: Transformers #45852 (one Stage 2 attempt), MLX-LM #1912 (two Stage 2 and three Stage 3 attempts), and MLX-LM #1971 (two Stage 2 attempts). Those are **eight prior inference attempts across the three Development families**. None produced an independently verified repair; the #1912 applied edit failed all 14 hidden cases. Failed and unchanged attempts still count as exposure.

The original split protocol, qualification catalog, source revisions, regression logs, and prior exposure ledger remain unchanged. The append-only [exposure disposition](../results/mlsys-task-deep-qualification-v1-exposure-append-v1.json) corrects only task-use eligibility: all three allocated Development tasks are exposed, so this split has **zero tasks available for fresh Development evaluation**. The audit did not run a model, revisit a task, or execute a held-out grader. Build a replacement Development pool from independently audited candidates; do not move these tasks into Train or Held-out.

Reproduce the reconciliation with `python scripts/audit_mlsys_deep_qualification_exposure_v1.py --stage3-ledger ../../outputs/VARE-stage3-2026-10-31/task-exposure-ledger.json --output work/exposure-audit-copy.json`. The script verifies the frozen qualification-catalog hash and the retained exposure-ledger hash before writing a new record.
