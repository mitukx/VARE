# GRPO group integrity and rollback reproduction packet

This packet joins two already frozen VARE-owned checks with the regression suite for complete group handling:

1. Rollout generation and replay preserve complete GRPO comparison groups under a nonmultiple target and bounded replay capacity.
2. A candidate-boundary failure after a real optimizer update restores the incumbent state.
3. A failure injected immediately after the pinned trainer's real optimizer mutation also restores model, optimizer, RNG, and next-rollout identity.

It is a reproduction aid, not a new experiment or evidence of policy improvement. The integration smokes use a tiny randomly initialized GPT-2 and one pinned RVL trainer revision. They do not measure learning quality, pretrained-model behavior, production deployment, or external reproduction.

## Requirements

- Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, pytest, CPU only.
- A local `src/rvl_systems` directory from RVL commit `c7e646b043cb56e5ea3c2623bb8a61e065451f72`. Both validators check the five source-file hashes before importing the trainer.
- No model weights, dataset, GPU, API, or network access is used by the run.

## Run

```bash
output_root=$(mktemp -d)
scripts/reproduce_grpo_integrity_packet.sh \
  /path/to/python \
  /path/to/Recursive-Verification-Lag/src/rvl_systems \
  "$output_root/run"
```

The `mktemp` directory gives each run a fresh parent; the script refuses to overwrite existing evidence. Replace the two machine-specific placeholders with an executable Python environment and the pinned RVL source tree. The script runs the focused group/replay/RVL-hook tests and both frozen real-trainer rollback smokes, retains each raw stdout/stderr log, and writes a SHA-256 manifest. A failed command leaves a `failure.json` and all logs produced before the failure. The original frozen protocols and prior run bundles remain unchanged.

## Source records

- [Group-integrity report](rollout-group-integrity-v1-report.md) and focused regression suite.
- [Candidate-boundary rollback report](rvl-grpo-partial-failure-report.md), [protocol](../protocols/rvl_grpo_partial_failure_v1.json), and [run bundle](../results/rvl-grpo-partial-failure-v1/run-1/).
- [In-step rollback report](rvl-grpo-midstep-fault-report.md), [protocol](../protocols/rvl_grpo_midstep_fault_v1.json), and [run bundle](../results/rvl-grpo-midstep-fault-v1/run-1/).

This packet reduces the commands needed to reconstruct the existing local evidence. A run by the repository author or CI is not an outside-person reproduction.
