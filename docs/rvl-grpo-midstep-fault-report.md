# RVL GRPO in-step optimizer-fault rollback validation v1

## Question and decision

If the pinned RVL trainer's real `train_step` raises after its optimizer has already changed model weights and optimizer state, does the VARE adapter restore the incumbent before returning the error?

**Pass for this injected boundary.** All 12 frozen checks passed. The actual pinned trainer's `train_step` ran without replacement. Its real `optimizer.step` completed, then a wrapper raised before `train_step` returned. Both model parameters and optimizer state had changed at the injection point. VARE propagated the error, restored the model, optimizer, and CPU RNG, preserved the active policy and counter, removed the failed candidate, and used the incumbent on the next rollout.

## Frozen setup

The protocol and validator were committed to the public repository at `63ba181ae54a67738b6d40730ce10a9e8bdd7ba0` before execution. The validator checked five pinned source-file hashes from RVL commit [`c7e646b`](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py).

The fixture was a randomly initialized 3,696-parameter GPT-2 with a 16-token vocabulary, one layer, 16 hidden units, and two heads. One prompt had two one-token responses with deterministic rewards. The entire run used a single CPU thread; no pretrained weights, dataset, CUDA, paid compute, or network access was used.

| Runtime measure | Result |
| --- | ---: |
| Python / PyTorch / Transformers | 3.12.12 / 2.9.1 / 4.57.3 |
| Device / PyTorch threads | CPU / 1 |
| Wall time | 3.476 s |
| CUDA initialized | No |
| Checks | 12/12 passed |

The immutable [protocol](../protocols/rvl_grpo_midstep_fault_v1.json), [validator](../scripts/validate_rvl_grpo_midstep_fault.py), and [run bundle](../results/rvl-grpo-midstep-fault-v1/run-1/) retain the procedure, hashes, and output. A later exact-commit rerun is in [run-3](../results/rvl-grpo-midstep-fault-v1/run-3/). [Run-2](../results/rvl-grpo-midstep-fault-v1/run-2/) is retained as a reproducibility failure: the validator accepted matching source-file hashes from a different Git commit. The revision guard rejects that case and dirty checkouts; its unit tests pass. This improves provenance checks but does not count as external reproduction.

### Clean-checkout reproduction

The validator requires Python 3.12.12, PyTorch 2.9.1, and Transformers 4.57.3;
it rejects other versions. VARE has no runtime dependencies, so install the two
experiment packages explicitly. The official PyTorch 2.9.1 install commands
provide a CPU-only wheel for Linux; macOS uses the regular wheel ([PyTorch
versioned commands](https://pytorch.org/get-started/previous-versions/)).

```bash
git clone https://github.com/mitukx/VARE.git
cd VARE
git checkout 63ba181ae54a67738b6d40730ce10a9e8bdd7ba0
python3.12 -m venv .venv
. .venv/bin/activate
python --version  # must print 3.12.12

# macOS:
python -m pip install torch==2.9.1
# Linux CPU (use this instead of the macOS line):
# python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install transformers==4.57.3
python -m pip install -e .
git clone https://github.com/mitukx/Recursive-Verification-Lag.git /tmp/Recursive-Verification-Lag
git -C /tmp/Recursive-Verification-Lag checkout c7e646b043cb56e5ea3c2623bb8a61e065451f72
python scripts/validate_rvl_grpo_midstep_fault.py \
  --rvl-source /tmp/Recursive-Verification-Lag/src/rvl_systems
```

The validator verifies the five expected upstream source hashes before loading
the trainer. It runs a random-initialized tiny model without dataset or model
downloads. A pass emits all 12 checks as `true` and `status: pass`; it fails if
the runtime versions, source hashes, CPU-only condition, or 120-second limit do
not match the frozen protocol. Package and source installation require network
access; the validation run itself sets model-hub and dataset access to offline
mode.

## Limits

This tests rollback after an optimizer update has completed but before the enclosing `train_step` returns. It does not inject a fault during an optimizer kernel, simulate process or machine loss, test another trainer revision, exercise pretrained-model learning, establish downstream task improvement, or provide an independent external reproduction. It complements the earlier [candidate-boundary smoke](rvl-grpo-partial-failure-report.md); the two checks cover distinct exception locations in one pinned integration.
