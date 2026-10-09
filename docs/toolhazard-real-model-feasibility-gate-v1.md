# ToolHazard real-model feasibility gate v1

**Decision: NO-GO before model scoring or training.** The candidate data split is real and disjoint, but this host cannot run the released post-training path, and its grader has not been independently audited. No model outputs or task scores were generated. This is a quantified resource blocker, not a task-performance result.

## Candidate screened

- Model: cached `Qwen/Qwen2.5-0.5B-Instruct`, snapshot `7ae557604adf67be50417f59c2c2f167def9a775`.
- Task: ToolHazard tool-selection indirect-prompt-injection tasks, repository revision `544b73b12a25431cb0be3eb43df41b4aacce5335`.
- The paper describes executable stateful environments and reports alignment gains on ToolHazard and AgentDojo; reproducing that direction would not itself be a novel method claim ([paper](https://arxiv.org/abs/2608.11878)).
- A pinned read-only audit found 329 unique RL training tasks over 17 task environments and 86 unique test tasks over 28 task environments. Task IDs and task-environment IDs do not overlap. The environment metadata IDs also do not overlap. The training release contains 193 trajectories. Exact file hashes and the reproduction command are in [`source_audit.json`](../results/toolhazard-feasibility-gate-v1/source_audit.json) and [`toolhazard_feasibility_gate_v1.py`](../reproducers/toolhazard_feasibility_gate_v1.py).

This establishes a plausible train/test source, not yet the requested train/dev/confirmation protocol. The 86 test tasks would still need a frozen environment-disjoint development/confirmation partition before any model output is opened. Task records contain `checklist_with_func` Python checker snippets; their semantics and execution safety have not been independently checked, so the independent-grader requirement is **not yet met**.

## Resource mismatch

The local machine has 32 GiB unified memory, one available Apple MPS device, and no CUDA devices. System Python has PyTorch 2.8.0, Transformers 4.57.3 and Accelerate 1.10.1; it lacks TRL, PEFT, OpenAI SDK and MLX. VARE's Python 3.12 environment lacks PyTorch, Transformers, TRL, Datasets and PEFT. A Qwen2.5-0.5B checkpoint is cached; the released ToolHazard configuration instead names a Qwen3-4B checkpoint.

The ToolHazard RL recipe requires the external ROLL framework and provides an eight-GPU configuration: `num_gpus_per_node: 8`, Megatron training with tensor parallelism 2, vLLM inference, BF16 and FlashAttention 2. It specifies 20 update steps, 64 environment groups × 8 samples per step (512 trajectories/step), up to 40 actions/trajectory, 4,096 generated tokens/action, and a 32,000-token trajectory length. That is **10,240 rollout trajectories per seed** and **30,720 for three RL seeds**. At the configured sequence-length ceiling, the three-seed RL arm represents up to **983,040,000 trajectory token positions**, before SFT, no-update evaluation, and test rollouts. These are configuration-derived workload counts, not measured runtime.

The released ROLL guide lists CUDA/cuDNN and vLLM as installation prerequisites; the ToolHazard recipe maps workers to devices 0–7. The available MPS device cannot satisfy that configuration. Hugging Face supports standard single-device MPS training, but documents that MPS does not support distributed training; that does not provide the missing ROLL/Megatron/vLLM agentic path ([ROLL guide](https://github.com/alibaba/ROLL/blob/main/docs_roll/docs/User%20Guides/Pipeline/agent_pipeline_start.md), [Transformers MPS documentation](https://huggingface.co/docs/transformers/perf_train_special)).

## What is not established

No base-model task success, action/output parse rate, reward variance, false acceptance, KL, or wall-clock rollout/update cost was measured. The official evaluator uses an OpenAI-compatible inference interface; this environment has no local endpoint or installed SDK. We did not install a substitute stack or write a new agent/runtime adapter because that would be the substantial missing component, and its correctness and MPS performance are unmeasured. Consequently no matched no-update/SFT/GRPO protocol was frozen and no training was started.

## Minimum route to reopen

The direct reproduction route needs a **free eight-CUDA-GPU allocation** compatible with the pinned ROLL stack, access to the configured model weights, and an independent audit of the released test checker before evaluation. The no-spend local route instead requires a deliberately smaller, single-MPS agentic execution and GRPO stack, followed by a one-update/save/reload smoke and a measured rollout-time gate; only after those pass should the 86 test tasks be split by environment into frozen development and untouched confirmation cohorts. That local port is currently unvalidated, so claiming it feasible would be premature.

Reproduce the source/split audit with:

```bash
python3 reproducers/toolhazard_feasibility_gate_v1.py
```
