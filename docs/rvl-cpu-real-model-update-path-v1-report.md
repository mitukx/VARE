# VARE → RVL CPU real-model update-path feasibility v1

**Status: incomplete artifact retention; do not count as a validated update-path result.**

## Question

Can the pinned VARE integration execute one CPU-only GRPO update on a cached real model, restore the incumbent, and save/reload the candidate within a fixed local resource limit? This is an execution-feasibility question only. It does not test task improvement, generalization, or capability.

## Run record

- Protocol: [`rvl_cpu_real_model_update_path_v1.lock.json`](../protocols/rvl_cpu_real_model_update_path_v1.lock.json), SHA-256 `9a29bc9daf9f67e529cbabe505a494bab0b60ba6a1eef625afc2e23047f3c2c5`.
- Source: VARE base revision `d8c72bac9188f33e18daefa1577d084f40f65cd3`; pinned RVL revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72`.
- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Raw retained generation record: [`progress.json`](../results/rvl-cpu-real-model-update-path-v1/run-1/progress.json).
- Same-host audit: [`audit.json`](../results/rvl-cpu-real-model-update-path-v1/run-1/audit.json), run with [`audit_rvl_cpu_real_model_update_path_v1.py`](../scripts/audit_rvl_cpu_real_model_update_path_v1.py).

## Findings and limitation

The process output reported a completed one-step CPU GRPO update and a matching saved/reloaded parameter fingerprint. The retained progress file contains five four-sample groups; a separate reward replay matched every stored label and verified that the first nonconstant group was the fifth prompt, with rewards `[0, 0, 1, 0]`. The protocol, runner, VARE adapter, and pinned RVL source hashes match.

The frozen runner did not write its terminal result to `summary.json` or finalize `progress.json`. The optimizer metrics and round-trip record therefore are not preserved in a complete machine-readable bundle. The independent audit marks this run incomplete rather than treating clipped process output as adequate retained evidence. The prompts and all generations are consumed and excluded from later experiments.

The run also emitted a Transformers tokenizer-regex warning while reloading the saved candidate. Since v1 did not compare token IDs before and after reload, tokenizer behavior is unverified even though the runner output reported a parameter fingerprint match.

## Decision

Do not use v1 as evidence that the full update/save/reload gate passed. Preserve the artifact-retention failure. The fresh-cohort v2 protocol and runner are frozen separately in [`rvl_cpu_real_model_update_path_v2.lock.json`](../protocols/rvl_cpu_real_model_update_path_v2.lock.json); v2 writes its terminal summary and checks tokenizer prompt IDs across save/reload. Neither version measures whether an update improves task success.
