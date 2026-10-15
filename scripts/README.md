# Reproduction and diagnostic entry points

Run commands from the repository root. Choose a fresh output directory for comparisons; preserve failed runs. This is a curated index, not a claim that every historical script runs with the default dependencies. See the [documentation index](../docs/README.md) for owning reports and study decisions.

| Question | Command / owning contract |
| --- | --- |
| Does the current default CPU suite pass? | `python -m pytest --ignore=tests/test_math500_grader_v1.py --ignore=tests/test_math500_grader_v2.py` |
| Do killed evaluation workers leave running graders? | `python scripts/reproduce_grader_owner_loss_v1.py --output /tmp/vare-owner-loss-new`; `python scripts/audit_grader_owner_loss_v1.py /tmp/vare-owner-loss-new` / [report](../docs/grader-owner-loss-v1-report.md) |
| Do checkpoint FIFO/replacement/growth faults fail closed and recover? | `python scripts/reproduce_checkpoint_read_bounds_v1.py --output /tmp/vare-checkpoint-io-new`; `python scripts/audit_checkpoint_read_bounds_v1.py /tmp/vare-checkpoint-io-new` / [report](../docs/checkpoint-read-bounds-v1-report.md) |
| Does trajectory capture reject premature completion and replay the dispatch URL policy? | `python scripts/reproduce_trajectory_capture_boundary_v1.py --output /tmp/vare-capture-new`; `python scripts/audit_trajectory_capture_boundary_v1.py /tmp/vare-capture-new` / [report](../docs/trajectory-capture-boundary-v1-report.md) |
| Can the finite four-member GRPO audit witness be independently enumerated? | `python scripts/reproduce_grpo_audit_independent_reproduction_v2.py --output /tmp/vare-grpo-audit-independent-reproduction-v2.json` / [report](../docs/grpo-audit-independent-reproduction-v2-report.md) |
| Are retained large blobs unchanged? | `PYTHONPATH=src python scripts/check_tracked_blob_budget.py` / [retention](../docs/artifact-retention.md) |
| Do sibling loops/crash restarts fail closed? | `python scripts/reproduce_runtime_round_ownership_v1.py --output /tmp/vare-round-review` / [report](../docs/runtime-round-ownership-v1.md) |
| Does interrupted publication avoid false success/reentry? | `python scripts/reproduce_promotion_publication_recovery_v1.py --output /tmp/vare-publication-review` |
| Does accepted publication retain round ownership across cancellation? | `python scripts/reproduce_promotion_cancellation_ownership_v1.py --output /tmp/vare-publication-cancel-review` / [report](../docs/promotion-cancellation-ownership-v1.md) |
| Does cancelled discard finish restoration? | `python scripts/reproduce_candidate_discard_recovery_v1.py --output /tmp/vare-discard-review` |
| Are evaluator siblings terminal before rollback? | `python scripts/reproduce_engine_evaluation_lifecycle_v1.py --output /tmp/vare-evaluation-review` |
| Does operator revocation retain cancellation-safe rollback? | `python scripts/reproduce_engine_evaluation_interlock_integration_v1.py --output /tmp/vare-interlock-review` |
| Does the actual CPU HF path preserve inference state/worker ownership? | `python scripts/reproduce_rvl_inference_transaction_v1.py --output /tmp/vare-inference-review` / [optional dependencies and report](../docs/rvl-inference-transaction-v1-report.md) |
| Can retained scheduler/recovery records be reconstructed? | `python scripts/audit_scheduler.py results/cpu-scheduler-v1`; `python scripts/audit_recovery.py results/cpu-recovery-v1` |
| Which historical grader can run on this platform? | [Reviewer guide](../docs/reviewer-guide.md), [CLI](../docs/execution.md); urllib3 calibration requires macOS |
| Which model study may run next? | [Current gaps](../docs/current-gaps.md), [experiment index](../docs/experiments.md), [agent charter](../AGENTS.md); existing consumed cohorts are not new confirmation data |

For the mathematical checker tests, install `requirements/math500-grader-test.lock.txt` and run their two test files separately. GPU notebooks, model-download scripts, and historical exploratory runs are not part of the default quick start. The optional runtime/trainer locks in `requirements/` document their own setup.

On a multiprocess evaluator-test failure, `VARE_TEST_DIAGNOSTICS_DIR=/tmp/vare-diagnostics` enables JSON capture. The existing lease, timeout, and acceptance criteria remain unchanged; CI uploads the file when present.
