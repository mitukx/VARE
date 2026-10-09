# TRL AsyncGRPO normalization PR #7249: clean-clone replay

**Outcome: pass.** A single command fetched fresh, clean checkouts of the pinned TRL base and PR head, reran the frozen full-Trainer CPU DDP experiment, checked its predeclared acceptance gates, and compared every summary field against the committed v4 summaries. All non-timing fields matched exactly. This improves replayability of the existing implementation finding; it is not new algorithmic evidence, outside human reproduction, or model-capability evidence.

## Why this replay

The v4 report records a real two-process CPU Gloo update through `Trainer.train()`, `Accelerator.backward`, DDP gradient reduction, and SGD. Its original command depended on local temporary source checkouts. This follow-up tests whether a reviewer can fetch the exact source revisions and recover the same result without those temporary directories.

## Frozen inputs and execution

The wrapper verifies the committed protocol digest and original v4 runner hash, then checks Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0, and Gloo availability. It fetches the base commit `a98fa6a4428f9aae58dfb26d729d7437f662f27a` and PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40` into temporary clean clones, verifies audited source-file hashes, and launches the unchanged v4 runner for each arm. No GPU, paid API, or external spending was used.

```bash
python scripts/reproduce_trl_pr7249_full_trainer_ddp_v4.py \
  --output results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-N
```

`run-N` must be a new output directory. The Python environment must already be installed; the wrapper checks versions but does not install packages. `--pythonpath-extra PATH` can be repeated to expose an existing dependency overlay to child processes.

## Result

The replay completed both two-process arms in 16.75 seconds total. Both source clones were clean and matched the frozen commits and source-file hashes. All four base and four candidate cases passed the v4 gate checks. The base again exceeded the defect threshold in the uneven-token cases (gradient errors `2.3206e-2` and `6.6667e-2`); the PR candidate again matched the pooled-token oracle within the frozen `1e-5` tolerance in every case.

The wrapper compared each replay summary with its committed v4 reference summary. All fields matched exactly except `elapsed_seconds`, which is expected to vary. The committed run bundle contains the manifest, comparison record, both summaries, and launcher logs under [`results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-3/`](../results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-3/). An earlier wrapper attempt failed before any trainer update because it pre-created an output directory that the frozen runner expected to create; that failure is retained in [`run-1/failure.json`](../results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-1/failure.json). The corrected initial replay is retained in `run-2/`; run-3 is the execution with the automated exact-summary comparison.

## Interpretation and limits

This is a same-author, same-host clean-source-clone replay. It demonstrates that the retained experiment can be reconstructed from pinned upstream source and that its previous numeric summary is stable under that replay. It is not an independent human reproduction or review. The synthetic fixture and manually supplied rank-local DataLoaders still bypass asynchronous rollout workers and production dataloader dispatch. The underlying PR is an upstream candidate, so VARE documents and validates it rather than duplicating its implementation.

## Decision

**Accept the reproduction result and stop extending this normalization branch absent a specific reviewer question.** It reduces the cost of external inspection but does not close the real-model task-success gap. The next high-value research action remains finding a genuinely CPU-feasible task/update path with independent held-out task success, or obtaining a technically independent review of a retained packet.

No full repository test suite was run; the frozen experiment itself and its gates were executed.
