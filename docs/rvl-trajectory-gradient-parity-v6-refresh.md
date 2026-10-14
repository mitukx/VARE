# RVL–VARE trajectory gradient parity v6: source-current refresh

**Status:** completed CPU reproducibility refresh. **Evidence tier:** E0 synthetic integration correctness. V6 repeats the v5 measurement against VARE main revision `3ca6d7e765a20fb85dc927c9adfec81f77c09a65`. It is a source-current evidence refresh, not a new method, scientific result, capability result, or RSI result.

## What was rerun

V5 already tested the intended parity property. Its lock pins the VARE files as they existed for that run, so its runner correctly rejects the later checkout. V6 changes the source revision and file hashes in a new lock and reruns the same seed (`20261014`), four-sample synthetic fixture, tiny 1,024-parameter CPU model, temperature (`0.8`), pinned RVL revision (`c7e646b043cb56e5ea3c2623bb8a61e065451f72`), trainer settings, independent materialized-target oracle, numeric tolerances, and prompt-merged negative control. V4 and v5 artifacts remain unchanged.

The v5 result does not record the VARE Git commit used for execution, so this report does not attribute one to that run. Its baseline source content is identified by the four `vare_source_files_sha256` entries in `protocols/rvl_trajectory_gradient_parity_v5.lock.json`. V6 records the current main revision and corresponding source hashes; this is a provenance refresh, not a claim that v5 ran at any particular commit.

This compares the actual pinned RVL `train_step` loss and parameter gradient against an independent autograd calculation over targets emitted by VARE's replay reader, group-relative advantages, and trajectory materializer. Two separate two-member groups intentionally share a prompt ID; response lengths are 1, 3, 2, and 4. Each group's materialized target weights sum to one. For a joint RVL batch, the oracle scales group-local target weights by `group_size / total_batch_size`, matching RVL's sample-mean reduction.

The optimizer step is intercepted as a counted no-op. The run checks that model parameters and buffers, optimizer state, CPU/Python RNG state, and module modes are unchanged. The prompt-ID-merged control is deliberately incorrect and is compared by gradient, since its scalar loss is close to zero too.

## Result

| Measurement | V6 |
|---|---:|
| VARE source revision | `3ca6d7e765a20fb85dc927c9adfec81f77c09a65` |
| RVL source revision | `c7e646b043cb56e5ea3c2623bb8a61e065451f72` |
| Runtime | macOS arm64, Python 3.12.12, PyTorch 2.9.1, CPU, 1 thread |
| Absolute loss difference | 1.4901161193847656e-08 |
| Maximum parameter-gradient absolute difference | 7.450580596923828e-09 |
| Prompt-merged control gradient difference | 0.0819641649723053 |
| Model/optimizer/RNG invariants | all passed |
| Parameter update | none; optimizer step intercepted once |
| Wall time | 1.54 s |

The measured values match v5's reported loss and gradient differences. The changed VARE source files since v5 are locked and verified in v6, but this does not extend the experiment's coverage: it remains a synthetic one-turn CPU objective/gradient interoperability check. It does not establish task success, policy improvement, real-model training behavior, reward authenticity, general multi-turn/tool behavior, or RSI.

## Reproduction and artifacts

Run from the VARE repository checkout at the locked revision, with the pinned RVL Git commit available locally:

```sh
PYTHONPATH=src python3.12 scripts/experiments/rvl_trajectory_gradient_parity_v6.py \
  --rvl-git /path/to/Recursive-Verification-Lag
PYTHONPATH=src python3.12 scripts/experiments/audit_rvl_trajectory_gradient_parity_v6.py
```

Both commands are offline and use CPU only. The runner verifies its own hash, the current VARE commit, locked VARE source hashes, pinned RVL revision, and six RVL source hashes. The auditor independently checks the result against the protocol lock, including the source-current refresh declaration and runner/auditor/source hashes.

- Protocol: `protocols/rvl_trajectory_gradient_parity_v6.lock.json`
- Runner: `scripts/experiments/rvl_trajectory_gradient_parity_v6.py`
- Auditor: `scripts/experiments/audit_rvl_trajectory_gradient_parity_v6.py`
- Result: `outputs/rvl_trajectory_gradient_parity_v6.json`
- Audit: `outputs/rvl_trajectory_gradient_parity_v6.audit.json`

The audit checks the recorded result and lock; rerunning the runner performs the numerical reproduction. This is local engineering evidence, not an external evaluation or an xAI/SpaceXAI MTS Eval result.
