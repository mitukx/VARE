# VARE → RVL CPU real-model update-path feasibility v3

**Measurement gates: satisfied in the retained record. Runner process: nonzero exit after the gates, due to a redundant cleanup error.** This is execution-feasibility evidence, not model-quality evidence.

## Frozen setup and artifacts

- Protocol: [`rvl_cpu_real_model_update_path_v3.lock.json`](../protocols/rvl_cpu_real_model_update_path_v3.lock.json), SHA-256 `722debadc2e37eea9e8c1745ce7614d6276dac454bb8c006e87ec06343a33c53`.
- VARE source revision: `644b436e54b2c0740e3f3df3eda7b8b4e7e5d859`; pinned RVL revision: `c7e646b043cb56e5ea3c2623bb8a61e065451f72`.
- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Retained run: [`summary.json`](../results/rvl-cpu-real-model-update-path-v3/run-1/summary.json), [`progress.json`](../results/rvl-cpu-real-model-update-path-v3/run-1/progress.json), and [`audit.json`](../results/rvl-cpu-real-model-update-path-v3/run-1/audit.json).

## Measurements

The first nonconstant four-sample group had rewards `[0, 1, 0, 0]`. VARE invoked one pinned RVL GRPO optimizer step. The record contains 290 finite, nonzero-gradient tensors; 291 parameter tensors changed, with maximum absolute parameter delta `1.9073486328125e-6`.

After the update, the adapter restored the incumbent's model weights, optimizer state, CPU RNG, and training mode exactly. The candidate was saved, reloaded on CPU, and its parameter fingerprint matched. The tokenizer produced identical token IDs on all eight frozen task prompts before and after save/reload. The same-host replay audit recomputed rewards and advantages and checked pinned source/model hashes: **19/19 checks passed**.

Wall time was **28.30 seconds** and peak RSS was **13.89 GB**, within the frozen limits of 1,200 seconds and 22 GiB. CPU was used; MPS was available but disabled. Transformers still emitted a tokenizer-regex warning. Exact IDs matched on these eight prompts, which does not establish tokenizer correctness for arbitrary text.

## Runner failure and evidence boundary

The runner exited nonzero after all measured fields had been written because cleanup attempted to delete `loaded_tokenizer` a second time. The audit identifies this as a post-gate `UnboundLocalError`. The frozen status remains `failed`; it has not been edited to `passed`. The specific execution question is supported by the retained measurements, while the runner's terminal handling still has a small defect.

This run does **not** measure reward effectiveness, held-out task success, policy improvement, generalization, or model capability. The temporary checkpoint was deleted after round-trip verification. The audit is a separate implementation on the same host, not outside reproduction.

## Decision

**CONTINUE**, with the research focus moving from “can a real CPU GRPO update execute?” to “can a frozen, fresh task show independent task-success improvement over strong matched baselines?” Do not reuse any v1–v3 smoke prompts. Before that experiment, choose a task/model pair with nontrivial base success and affordable multi-seed training, then freeze the full comparison and evaluation protocol. Preserve the runner cleanup defect as part of v3's record; no post-hoc rerun of this cohort.
