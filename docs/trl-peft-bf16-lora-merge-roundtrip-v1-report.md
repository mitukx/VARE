# PEFT BF16 LoRA merge/unmerge roundtrip v1 — report

**Decision: STOP as a research direction; retain as a local reproduction of a known upstream finding.**

## Question and scope

Does PEFT 0.19.1 mutate a BF16 base weight when its LoRA adapter is merged and immediately unmerged, the operation TRL uses while synchronizing weights to vLLM? This CPU-only check isolates the actual PEFT layer implementation using the pinned Qwen2.5-0.5B-Instruct first-block `q_proj` tensor and a deterministic synthetic nonzero adapter. It does not run TRL, vLLM, training, a learned adapter, or a downstream task.

The locked protocol is [trl_peft_bf16_lora_merge_roundtrip_v1.lock.json](../protocols/trl_peft_bf16_lora_merge_roundtrip_v1.lock.json). The implementation and raw output are in [run.py](../experiments/trl_peft_bf16_lora_merge_roundtrip_v1/run.py) and [results.json](../experiments/trl_peft_bf16_lora_merge_roundtrip_v1/results.json). The model tensor is from revision `7ae557604adf67be50417f59c2c2f167def9a775`; its safetensors shard SHA-256 is `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. PEFT 0.19.1 wheel SHA-256 is `2113f72a81621b5913ef28f9022204c742df111890c5f49d812716a4a301e356`. The inspected TRL source was commit `f4526e10e25c8618855932528c232e2284c7801c`; the `vllm_generation.py` file SHA-256 was `01a67466494fe3618c6b02d15eac0fe10ebd9a1a355862bd32064a64771d48ef`.

## Results

The q_proj tensor has 802,816 BF16 elements. PEFT's actual `merge_adapter()` / `unmerge_adapter()` methods were called 100 times for each relevant arm.

| Arm | Changed elements after 1 cycle | Changed after 100 cycles | Max abs drift after 100 | Relative L2 drift after 100 | Fixed-input output delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16, unsafe merge, nonzero adapter | 53,597 | 53,597 | 0.00390625 | 6.52e-4 | 0.03125 |
| BF16, `safe_merge=True`, nonzero adapter | 50,596 | 50,596 | 0.00390625 | 4.96e-4 | 0.03125 |
| BF16, zero-B control | 0 | 0 | 0 | 0 | 0 |
| BF16, no merge control | 0 | 0 | 0 | 0 | 0 |
| FP32, unsafe merge | 19,311 | 19,311 | 2.98e-8 | 3.99e-9 | 9.54e-7 |

The BF16 unsafe arm first changed the base tensor on cycle 1; its L2 drift rose from `0.02969` to `0.03900` by cycle 100 while the changed-element count stayed fixed. The safe-merge arm reached the same measured drift at cycle 1 and stayed there through cycle 100. FP32 also was not bitwise exact, but its maximum drift and output delta were much smaller in this run. These observations are for one layer, one seed, and one synthetic adapter scale.

Run environment: Python 3.12.12, PyTorch 2.9.1, PEFT 0.19.1, CPU, about 2 seconds total. Peak RSS reported by the process was about 1.32 GiB, below the 2 GiB protocol ceiling. Command:

```sh
PYTHONPATH=/tmp/vare-peft-019 python experiments/trl_peft_bf16_lora_merge_roundtrip_v1/run.py \
  --model-file /path/to/cached/model.safetensors \
  --output experiments/trl_peft_bf16_lora_merge_roundtrip_v1/results.json
```

No repository test suite or CI was run for this report. The exact experiment is reproducible only where the pinned model shard is already cached and the pinned PEFT wheel is available; the script intentionally performs no download.

## Novelty and interpretation

This is **not a novel defect discovery**. TRL [issue #6688](https://github.com/huggingface/trl/issues/6688) was already open with the same concern. Before this run, a commenter had reported PEFT-level BF16 drift, an explanation for the GPU-side FP32/BF16 operand asymmetry, comparisons across 1–1,000 cycles, safe-merge and out-of-place controls, and a separate user had reproduced drift through actual TRL weight sync on Qwen2.5-0.5B-Instruct. That upstream evidence is broader and more directly relevant than this isolated CPU layer check. At the time of inspection the issue remained open and showed no linked PR.

This local run independently corroborates the narrow statement that a BF16 PEFT roundtrip can mutate a real pretrained base tensor under CPU PEFT's arithmetic path. CPU PEFT casts its delta back to the adapter dtype, so these measurements do **not** reproduce the GPU-side asymmetric-delta accumulation mechanism reported upstream. The safe-merge result also confirms that safe merge does not restore the original base bitwise, even though it bounded drift in this specific run. No claim is made about training divergence, checkpoint quality, task success, or prevalence in production.

## Research decision

Do not submit a duplicate issue or patch for this mechanism. Do not treat this reproduction as VARE novelty or systems impact. It adds a pinned, low-cost CPU confirmation on a real base tensor, but the novelty gate fails and the upstream thread already contains more complete end-to-end evidence. Return to the portfolio's unresolved priorities: independent outside review/reproduction of a retained result, or a materially distinct trainer defect with a production-path reproducer. The model-improvement evidence gap remains unchanged: no independently confirmed task-success improvement after a real policy update.
