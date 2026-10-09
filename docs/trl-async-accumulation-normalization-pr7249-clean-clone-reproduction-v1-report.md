# TRL AsyncGRPO normalization PR #7249: clean-clone replay

**Outcome: replay succeeded, but the underlying v4 protocol is a non-pass.** A single command fetched fresh, clean checkouts of the pinned TRL base and PR head and reproduced the committed summaries. A later audit found that v4's primary unequal-token fixture transposed its rank and microbatch axes: the lock specified totals `[13, 10]`, while the replayed summaries contain `[9, 14]`. Thus the replay confirms only that the alternate executed fixture is repeatable, not that the locked case passed.

## Why this replay

The v4 report records a real two-process CPU Gloo update through `Trainer.train()`, `Accelerator.backward`, DDP gradient reduction, and SGD. Its original command depended on local temporary source checkouts. This follow-up tests whether a reviewer can fetch the exact source revisions and recover the same result without those temporary directories. The later fixture audit limits its scientific interpretation as described above.

## Frozen inputs and execution

The wrapper verifies the committed protocol digest and original v4 runner hash, then checks Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, Accelerate 1.12.0, and Gloo availability. It fetches the base commit `a98fa6a4428f9aae58dfb26d729d7437f662f27a` and PR head `5234eb7c70f4ca7eb92fe8e01611a33eaff17f40` into temporary clean clones, verifies audited source-file hashes, and launches the unchanged v4 runner for each arm. No GPU, paid API, or external spending was used.

```bash
python scripts/reproduce_trl_pr7249_full_trainer_ddp_v4.py \
  --output results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-N
```

`run-N` must be a new output directory. The Python environment must already be installed; the wrapper checks versions but does not install packages. `--pythonpath-extra PATH` can be repeated to expose an existing dependency overlay to child processes.

## Result

The replay completed both two-process arms in 16.75 seconds total. Both source clones were clean and matched the pinned commits and file hashes. The recorded thresholds pass for the executed fixtures, but the primary unequal-token case does not match the locked layout. The base exceeded the threshold for the alternate uneven case (gradient error `2.3206e-2`) and the correctly shaped zero-local-token case (`6.6667e-2`); the candidate matched both actual fixtures to the oracle. These values are historical observations, not v4 frozen-protocol confirmation.

The wrapper compared each replay summary with its committed v4 reference summary. All fields matched exactly except `elapsed_seconds`, which is expected to vary. The committed run bundle contains the manifest, comparison record, both summaries, and launcher logs under [`results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-3/`](../results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-3/). An earlier wrapper attempt failed before any trainer update because it pre-created an output directory that the frozen runner expected to create; that failure is retained in [`run-1/failure.json`](../results/trl-async-pr7249-full-trainer-ddp-v4-clean-clone/run-1/failure.json). The corrected initial replay is retained in `run-2/`; run-3 is the execution with the automated exact-summary comparison.

## Interpretation and limits

This is a same-author, same-host clean-source-clone replay. It demonstrates that the retained alternate fixture can be reconstructed from pinned upstream source and that its numeric summary is stable under that replay. It is not an independent human reproduction or review. The synthetic fixture and manually supplied rank-local DataLoaders still bypass asynchronous rollout workers and production dataloader dispatch. The underlying PR is an upstream candidate, so VARE documents and validates it rather than duplicating its implementation.

## Decision

**Do not count this replay as validation of the v4 protocol.** Preserve the operational replay record and use the corrected v5 protocol for the locked comparison. This does not close the real-model task-success gap or provide outside review.

No full repository test suite was run; the frozen experiment itself and its gates were executed.
