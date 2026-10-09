# VARE → RVL CPU real-model update-path feasibility v2

**Result: failed before the tokenizer/model reload gate.** The run is retained and does not count as a passed update-path feasibility result.

## Frozen setup

- Protocol: [`rvl_cpu_real_model_update_path_v2.lock.json`](../protocols/rvl_cpu_real_model_update_path_v2.lock.json), SHA-256 `931805371d1d4b1278f73ce69f367c5363a3d51fd3dc348e5e264249ba9f02e7`.
- Source revision: `fdf0f6f9859a4d250cda709816c0670b059017d0`; pinned RVL: `c7e646b043cb56e5ea3c2623bb8a61e065451f72`.
- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Runner: [`run_rvl_cpu_real_model_update_path_v2.py`](../scripts/run_rvl_cpu_real_model_update_path_v2.py).
- Retained record: [`summary.json`](../results/rvl-cpu-real-model-update-path-v2/run-1/summary.json), [`progress.json`](../results/rvl-cpu-real-model-update-path-v2/run-1/progress.json), and [`audit.json`](../results/rvl-cpu-real-model-update-path-v2/run-1/audit.json).

## Evidence

The CPU GRPO call executed once on four samples from the first nonconstant group. The recorded finite-gradient and candidate-change checks passed; the adapter's incumbent model, optimizer, RNG, and training mode were restored. The attempt stayed within the frozen caps (27.47 seconds and 12.51 GB peak RSS).

The run then failed with `NameError` in the newly added tokenizer round-trip check: the pre-save `tokenizer` variable had been deleted before the check. The candidate checkpoint had been written to a temporary directory, but neither tokenizer comparison nor model reload completed. The same-host auditor independently replayed response rewards and matched the pinned source and protocol hashes. It verifies a faithful failure record, not a successful experiment.

Transformers also emitted a tokenizer-regex warning during reload. The v2 failure occurred before token-ID equivalence could be measured. Treat tokenizer round-trip behavior as unresolved.

## Decision

The fix is narrow: keep the pre-save tokenizer alive through the frozen-prompt token-ID comparison, then release it. V2 prompts are consumed and excluded. The fresh-cohort [`v3 protocol`](../protocols/rvl_cpu_real_model_update_path_v3.lock.json) freezes that correction before execution. No version in this sequence measures policy quality, downstream task success, or model capability.
