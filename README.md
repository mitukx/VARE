# VARE

VARE is a CPU-first outer control plane for post-training experiments: audited training signals, policy/verifier freshness, independent evaluation, and candidate promotion or recovery. It integrates with [Recursive-Verification-Lag (RVL)](https://github.com/mitukx/Recursive-Verification-Lag) for trainer/rollout machinery rather than duplicating a full RL framework.

**Research question:** Under a fixed compute budget, which post-training signal improves an independently measured held-out outcome, and how do we know the update and evaluation are correct?

**Current evidence:** No retained real-model update shows an independently confirmed held-out task-success gain. Training reward, a passing base gate, and a successful optimizer smoke are not that result.

## Start here

| Goal | Entry point |
| --- | --- |
| Review representative work and its limits | [Reviewer guide](docs/reviewer-guide.md) |
| Find code, contracts, reports, and experiments | [Documentation index](docs/README.md), [repository layout](docs/repository-layout.md) |
| Reproduce a bounded correctness result | [Script index](scripts/README.md) |
| Check the next admissible research step | [Current gaps](docs/current-gaps.md), [agent charter](AGENTS.md) |
| Inspect the complete retained history | [Experiment index](docs/experiments.md), [evidence ledger](docs/evidence.md), [roadmap](docs/roadmap.md) |

## Representative systems evidence

| Boundary | Measured scope |
| --- | --- |
| [Full-round runtime ownership](docs/runtime-round-ownership-v1.md) | Eight fixed CPU fault cases fail on pinned main; the correction passes all ten cases including two controls. Same-hooks loops share ownership/quarantine; SQLite reservations survive process death. Single-host trusted fixtures, not distributed atomicity. |
| [Publication acknowledgment](docs/promotion-publication-recovery-v1.md) and [owned discard](docs/candidate-discard-recovery-v1.md) | Frozen baseline/fix comparisons reject false promotion success and retain restoration through cancellation. These are correctness results, not learning gains. |
| [RVL inference transaction](docs/rvl-inference-transaction-v1-report.md) | CPU fixture and tiny GPT-2/HF checks preserve RNG/module modes and drain cancelled generation workers. No production prevalence or accelerator result. |
| [Historical urllib3 grader](docs/urllib3-3352-real-model-feasibility-v1-report.md) | Pinned failing source and public known fix distinguish the actual regression; protected-asset checks and task identity are retained. Two local model attempts produced no patch. Public task is retired from confirmation. |
| [Trajectory/replay contract](docs/trajectory-reference-v1.md) | CPU reference and RVL bridge retain complete group identity and token/verifier provenance. Pending/invalid reward is distinct from valid zero. |

A retrospective audit of the pre-existing [format-versus-correctness v2 record](docs/format-comparison-record-v1-report.md)
preserves the primary interval including zero and finds no active format-reward
signal in its retained artifacts. The original run's executor, source revision,
and hardware remain unverified; this is record analysis, not a new model run.

The additive [seed-aware summary v3](docs/seeded-summary-v3-report.md) keeps the
frozen v2 primary decision intact and distinguishes pass@4 within a fitted run
from pooled outputs across runs. Synthetic baseline/fix evidence also checks
that metrics and progress hashes describe the same captured bytes. This is
evaluation-correctness evidence, not a model-learning result.

## Retained model outcomes

| Study | Observation | Decision / limit |
| --- | --- | --- |
| [Exact-integer v1](docs/heldout-exact-integer-base-gate-v1-result.md), SmolLM2-360M-Instruct / mlx | Strict pass@4: 0/64 | Retired `stop_zero`; do not tune the parser to pass the observed cohort. |
| [Exact-integer v2](docs/heldout-exact-integer-base-gate-v2.md), Qwen2.5-3B-Instruct 4-bit / mlx | Strict pass@4: 29/64 (45.31%) | Base feasibility only; separate [one-update smoke](docs/heldout-exact-integer-update-cost-smoke-v1.md) retained save/reload and resource measurements, not a learning gain. |
| [OpenBookQA confirmation](docs/openbookqa-qwen25-grpo-sft-confirmation-v1-report.md), Qwen2.5-0.5B | Base 45.31%; matched SFT 33.33%; GRPO 0/128 on all three seeds under the frozen parser | Pairing retired; no general conclusion about GRPO. |
| ARC-Challenge, Qwen2.5-0.5B | GRPO 38.84% versus SFT 40.00% | Frozen gate failed; see the [experiment index](docs/experiments.md). Pairing retired. |
| [Format/correctness v2](docs/format-vs-correctness-comparison-v2.md) | Correctness RL minus format SFT: +0.46875 points, 95% CI [-0.078125, +1.015625] | Retrospective audit of a pre-existing record; primary non-pass and original execution provenance incomplete. No new training occurred. |

Negative results, consumed cohorts, protocol locks, and raw records retain their existing paths. A fresh study needs independently justified task/model/update feasibility, matched controls, multiple seeds, and untouched confirmation tasks.

## CPU quick start

Python >=3.11, from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest --ignore=tests/test_math500_grader_v1.py --ignore=tests/test_math500_grader_v2.py
PYTHONPATH=src python scripts/check_tracked_blob_budget.py
python scripts/reproduce_runtime_round_ownership_v1.py --output /tmp/vare-runtime-round-review
```

Use a new output directory. The comparison needs Git access to its pinned baseline. Optional mathematical graders and real trainer/model paths have separate dependency locks; follow their reports. Historical urllib3 calibration requires macOS. `minigrpo/`, `selfimprove/`, and accelerator notebooks are experimental paths with no retained model-quality gain; see the [layout](docs/repository-layout.md).

Runtime recovery requires old owners/workers to be terminal and state to be independently verified; [operator procedure](docs/runtime-round-ownership-v1.md#operator-recovery). Durable evaluator jobs use a separate [lease/recovery contract](docs/recovery.md). Neither runner is a hostile-code security sandbox.

The 235 tracked blobs >=1 MiB are frozen. New large artifacts stay outside Git with manifest hashes; [artifact retention](docs/artifact-retention.md). Published history is not rewritten. [License](LICENSE): Apache-2.0.
