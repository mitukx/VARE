# GSM8K DPO margin audit: clean-clone reproduction v1

## Question

Can a fresh checkout of the committed VARE source reproduce the existing adapter-to-margin audit using only its retained public bundle and an already-cached model, with no training or network fetch?

This checks checkout-level reproducibility. It is not an outside-person reproduction and does not reopen the consumed GSM8K cohort as new confirmation.

## Frozen protocol and execution

Protocol [`cpu_lm_gsm8k_dpo_margin_clean_clone_reproduction_v1`](../protocols/cpu_lm_gsm8k_dpo_margin_clean_clone_reproduction_v1.lock.json) was committed at `7a25c78` before execution. It pins the clean-clone commit, reproducer, prior forensic lock, input bundle manifest, reference output, model file hashes, runtime versions, resource limits, and pass criteria.

The clone was created locally from commit `7a25c787b6c587b70ee897c299c10c7ddc8f3e03` with `git clone --local --no-hardlinks`. Before inference, the clone was clean; package versions and model, source, protocol, and bundle hashes matched the lock; and the explicit output directory did not exist. The replay used Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, NumPy 2.4.4, CPU float32, the existing Hugging Face cache, and offline flags.

The correct replay form from the repository root is:

```sh
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/audit_gsm8k_dpo_margin_reconstruction_v4.py \
  --output /tmp/vare-gsm8k-margin-clean-clone-v1-run
```

The output directory must be new. The reproducer defaults to the committed `run-1` path and refuses to overwrite an existing directory, so the original `run-1/command.txt` is a historical first-run command, not a repeatable command after the bundle has been checked in. The v1 lock and command above correct that issue without changing the reproducer or its thresholds.

## Result

| Check | Clean-clone run | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Reproducer status / exit code | pass / 0 | pass / 0 | Pass |
| Maximum base-margin error | `4.19616699e-5` | `5e-5` | Pass |
| Maximum updated-margin error | `4.20212746e-5` | `5e-5` | Pass |
| Maximum metric/bootstrap error | `2.71952892e-7` | `5e-5` | Pass |
| Wall time / peak RSS | `152.73 s` / `3.79 GB` | `3600 s` / `6 GiB` | Pass |
| Clone status after replay | clean | clean | Pass |

The reconstructed per-seed held-out NLL changes and bootstrap interval match the earlier run within the frozen criteria. No training, tuning, row selection, package installation, paid compute, or network fetch occurred. The clean-clone bundle contains the protocol and reproducer snapshots, command, runtime/preflight record, stdout/stderr, exit code, summary, and a SHA-256 manifest.

## Decision and claim limits

**The adapter-to-margin reconstruction now passes from a fresh local clone on the same host and runtime.** The earlier run's command was not repeatable after committing its default output directory; that usability defect is documented and the explicit fresh-output command has now been exercised successfully.

This remains same-author, same-host, same-cache evidence on the same consumed rows. It is not independent human reproduction, optimizer/gradient verification, a new confirmation, a task-success gain, or a model capability result. Reproduction on another machine additionally requires the pinned packages and local model files whose hashes are in the protocol; model weights are not vendored.

The GSM8K cohort stays closed. The next meaningful check is a review/reproduction by a person outside this run, or a materially distinct model/task/update route that first passes the frozen viability gates. Until then, the main real-model task-success gap remains open.

## Artifacts

- Frozen [protocol](../protocols/cpu_lm_gsm8k_dpo_margin_clean_clone_reproduction_v1.lock.json).
- [Clean-clone run bundle](../results/cpu-lm-gsm8k-dpo-margin-clean-clone-reproduction-v1/run-1/).
- Original [forensic reconstruction report](cpu-lm-gsm8k-dpo-margin-reconstruction-v4-report.md).
