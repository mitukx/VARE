# Qwen2.5 / DeepMath / MATH-500 CPU-first feasibility package

**Decision:** GO only for a later, separately authorized, inference-only one-GPU base gate. NO-GO for training today. No GPU, MPS, model inference, or paid API was used in this package.

This is a proposed small engineering-research replication of established GRPO, not a novel algorithm claim. The question is whether 64 steps of standard GRPO on a small, fixed DeepMath cohort can improve independent MATH-500 exact-answer success over no update and same-data SFT. Nothing in the current evidence says that it will.

## Selected pairing

- **Model:** Qwen/Qwen2.5-0.5B-Instruct, Hugging Face snapshot 7ae557604adf67be50417f59c2c2f167def9a775; weights SHA-256 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe.
- **Task and RL data:** 256 hash-selected prompts from trl-lib/DeepMath-103K train, revision 066c50a88d4e14cefc056e31111db2dba17f6c68; pinned parquet SHA-256 e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7.
- **Training method:** standard TRL GRPOTrainer with accuracy_reward; LoRA on q_proj and v_proj, 4 sampled outputs per prompt, 64 optimizer steps, beta=0, and a 1,024-token completion cap.
- **Independent task metric:** greedy Pass@1 exact answer on HuggingFaceH4/MATH-500, revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be, test file SHA-256 35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132. The checker extracts only the last balanced \boxed{...}, parses LaTeX with latex2sympy2-extended, and compares values through an independent SymPy path. Missing, empty, or malformed final boxes fail closed.

This exact model / dataset / reward combination appears in TRL's official GRPO quickstart. The documentation says its example uses eight GPUs for about one day; it does not establish that this reduced protocol improves held-out task success. Its nearby reward curves used Qwen2-0.5B, with Qwen2.5 described as qualitatively similar. The DeepMath paper reports gains from its larger 7B DeepMath series, not this 0.5B experiment. A public one-GPU TRL report on the exact 0.5B / DeepMath pair found that 1,024-token completions restored reward signal relative to a 256-token default, but this is training-reward evidence, not independent capability evidence. We therefore pin 1,024 tokens and measure the actual base policy before asking to train.

## Freshness, data quality, and leakage checks

The exact model / task / evaluator combination is new in VARE's retained study records. Qwen2.5-0.5B-Instruct appears in prior VARE studies on other tasks; those cohorts and the retired StrategyQA, SVAMP, ARC, ToolHazard, and other pairings are not reused. DeepMath and MATH-500 do not appear in the prior study inventory as a VARE training / confirmation pair.

CPU audit results:

- MATH-500 has 500 unique IDs, 500 nonempty references, and no repeated normalized question. Its original levels are 1:43, 2:90, 3:105, 4:128, 5:134.
- DeepMath's TRL mirror has 97,870 rows and 96,735 unique normalized prompts. Deduplicate 1,135 excess rows; exclude 3 duplicate-prompt answer conflicts and 54 duplicate groups whose reward labels cannot be parsed. This leaves 96,678 eligible prompts.
- Exact normalized question overlap is zero both against the complete 97,870-row training mirror and against all 103,022 rows of the pinned original DeepMath source. The source audit found 101,744 unique normalized questions and 1,278 duplicate excess rows. All 288 selected base-gate / training prompts map back to the original source.
- The final answer labels in the TRL mirror and original source match for all 288 selected prompts after removing math delimiters and whitespace. This guards against a transformation or source-revision mismatch in the selected subset.
- The DeepMath authors report benchmark decontamination, including MATH-500. This package's exact-string audit confirms no remaining exact overlap. It did not run a semantic-neighbor audit, and it cannot rule out exposure in Qwen pretraining or instruction tuning.
- MATH-500 is a public benchmark. The 100 development IDs and 400 final confirmation IDs are hash-disjoint and stratified by original level. All references were read for checker self-tests, overlap auditing, and split construction. No model outputs or confirmation scores were generated or used for tuning. Treat the confirmation partition as reserved for this protocol, not as a hidden benchmark.

The original DeepMath R1 trajectories were checked as a candidate stronger SFT target and rejected: each of the three traces parsed on all 256 selected training prompts, but only 148/256 (57.8%) ended with the dataset's exact final answer for each trace column. The shortest correct trace was at most 2,048 Qwen tokens for only 17/148 prompts (11.5%). Unfiltered trace SFT would train on many incorrect or truncated solutions. The frozen SFT control therefore uses the same 256 verified labels, rendered as a short final \boxed{answer} target. It is a clean answer-supervision control, not a claim to be a stronger reasoning-distillation baseline; that limitation remains explicit.

## Independent checker validation

The checker does not call TRL's accuracy_reward and does not consume training reward values. Training and evaluation have distinct extraction and comparison code; both rely on latex2sympy2-extended for low-level LaTeX parsing, so parser-library errors are a residual shared risk.

On the pinned MATH-500 file, the CPU audit established:

- 500/500 reference answers score themselves.
- 500/500 item-specific zero/one-answer negative controls are rejected.
- 492/500 references parse symbolically; the remaining 8 use strict normalized-text fallback.
- The frozen level-stratified partition is 100 development / 400 confirmation, with no ID overlap.
- The tests include wrong, malformed, and empty final boxes after an earlier correct box. Only the final box is considered, so earlier scratch work cannot rescue an invalid final answer.
- An end-to-end CLI self-check over the frozen 100-item development partition scored synthetic reference answers 100/100. This validates evaluator wiring only; it is not a model result. Before the parser fix, a valid earlier box followed by an unbalanced final box incorrectly returned the earlier answer; it now fails closed.

Run the CPU checker tests and data audit from the repository root:

    python3.12 -m pip install --target /tmp/vare-math-verify-0.9.0 -r requirements/math500-study-cpu.lock.txt
    PYTHONPATH=.:/tmp/vare-math-verify-0.9.0 .venv/bin/pytest tests/test_math500_grader_v1.py -q
    PYTHONPATH=.:/tmp/vare-math-verify-0.9.0 python3.12 -m scripts.validate_math500_study_v1 --output /tmp/vare-math500-cpu-audit.json
    PYTHONPATH=.:/tmp/vare-math-verify-0.9.0 python3.12 -m scripts.audit_deepmath_source_pairing_v1 --output /tmp/vare-deepmath-source-audit.json

Expected audit invariants are recorded in the protocol lock. No dataset payload is committed; download the pinned source revisions and verify their SHA-256 values before running the validators.

For example, the required files can be fetched without loading model weights:

    curl -L https://huggingface.co/datasets/trl-lib/DeepMath-103K/resolve/066c50a88d4e14cefc056e31111db2dba17f6c68/data/train-00000-of-00001.parquet -o artifacts/math-grpo-cpu-first-2026-10-11/data/deepmath-train.parquet
    curl -L https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl -o artifacts/math-grpo-cpu-first-2026-10-11/data/math500.jsonl
    SOURCE_DIR=artifacts/math-grpo-cpu-first-2026-10-11/data/deepmath-original
    mkdir -p "$SOURCE_DIR"
    for shard in 00000 00001 00002 00003 00004 00005 00006 00007 00008 00009; do
      curl -L "https://huggingface.co/datasets/zwhe99/DeepMath-103K/resolve/5cf055d1fe3d7a2eb19719ac020211469736ae44/data/train-${shard}-of-00010.parquet" -o "$SOURCE_DIR/train-${shard}-of-00010.parquet"
    done

## First GPU gate: inference only

The current machine has no CUDA device. This turn did not use CUDA, MPS, or a model forward pass. CPU tokenization of the 288 selected prompts with the pinned local tokenizer yielded 51 minimum / 95 median / 180 p95 / 405 maximum prompt tokens; none exceeds the 2,048-token input cap. Selected reference answers are short (median 4 tokens, maximum 20). These numbers establish prompt and target shape, not model behavior or accelerator feasibility.

If later authorized, the first accelerator command is:

    python3.11 -m scripts.run_qwen25_deepmath_base_gate_v1 \
      --model-dir /path/to/Qwen2.5-0.5B-Instruct/snapshot \
      --data-dir artifacts/math-grpo-cpu-first-2026-10-11/data \
      --output artifacts/math-grpo-cpu-first-2026-10-11/gpu-base-gate.json \
      --max-wall-seconds 7200

The command fails closed unless it runs Python 3.11.2, sees exactly one CUDA GPU with at least 16 GiB VRAM, and finds the pinned model/data hashes and runtime versions in requirements/math500-study-gpu-gate.txt. It evaluates 32 TRAIN-only prompts × 4 stochastic outputs (128 outputs total), temperature 0.7, top-p 0.95, 1,024 maximum new tokens, and the fixed boxed-answer instruction. MATH-500 is not loaded. The run has a hard cap of 131,072 generated tokens and two hours; it performs zero optimizer updates and records peak allocated/reserved VRAM, throughput, answer format, independent success, actual TRL reward, and within-group reward variance.

The frozen base gate passes only if independent success is between 5% and 90%, at least 90% of outputs have a valid final box, at least 6 of 32 groups have mixed rewards, mean within-group reward variance is positive, reserved memory stays at or below 14 GiB and 90% of device memory, and the run finishes within two hours. Any failure retires this exact pairing without prompt edits or reuse of the 32 gate prompts. This is only permission to request one optimizer-step / save-reload resource smoke; it does not authorize multi-step training.

## Conditional study and compute estimate

If the base gate passes and the one-step smoke later fits, the locked comparison is:

| Arm | Seeds | Update |
|---|---:|---|
| No update | shared base | none |
| Answer-only SFT | 3 | 256 same prompts × 4 epochs; LoRA r=8, alpha=16, q/v projections; LR 2e-4 |
| Standard GRPO | 3 | 64 optimizer steps; per-device batch 1, accumulation 4, 4 generations; LoRA r=8, alpha=16, q/v projections; LR 1e-6; beta=0; max completion 1,024 |

The three GRPO seeds generate 1,024 completions each (3,072 total; at most 3,145,728 generated tokens). SFT presents 3,072 labeled examples across its three seeds. The fixed development gate uses 700 greedy completions across the shared base and six updated candidates. If it passes, final confirmation uses 2,800 greedy completions (shared base plus six updated candidates over 400 IDs). These are ceilings implied by the protocol, not measured runtimes. The GPU feasibility gate and one-step smoke must supply actual tokens/second, update time, checkpoint time, and memory before a full-run wall-clock budget is approved.

Development advances only if mean GRPO success is at least 5 percentage points above base, no lower than SFT, and at least 2/3 GRPO seeds improve over base. Confirmation success requires at least a 5-point GRPO improvement over both base and SFT, paired level-stratified bootstrap 95% lower bounds above zero for both differences, and at least 2/3 seeds better than each control. Failure stops the study. Report per-level exact success, format rate, reward / task-success disagreement, per-seed variance, KL to base, runtime, and peak memory.

## Current limits and decision

**GO:** CPU package and checker are ready; this exact pair merits one future inference-only CUDA gate because the trainer and data are in the official recipe and published DeepMath results support the dataset family.

**NO-GO:** No base success rate, base reward variance, CUDA throughput, VRAM, optimizer feasibility, update, or independent capability gain has been measured. No multi-seed experiment is authorized or justified yet. The full study's wall-clock and hardware budget remain unknown until the gate and one-step smoke run.

A passing gate would warrant asking to run one optimizer step with save/reload and collect actual resource measurements. Only after that evidence would we request authorization for the matched multi-seed comparison.

## Sources

- [TRL GRPO trainer quickstart](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md) — exact model / dataset / reward example and stated 8-GPU runtime; the plotted curves use Qwen2-0.5B.
- [TRL issue #5697](https://github.com/huggingface/trl/issues/5697) — one-GPU report for the exact Qwen2.5-0.5B / DeepMath pairing, including completion truncation, reward, and VRAM measurements at different token caps. The issue was closed after the default was changed; it motivates the explicit token cap, not a claim of a current bug.
- [DeepMath-103K paper](https://arxiv.org/abs/2504.11456) and [pinned dataset revision](https://huggingface.co/datasets/zwhe99/DeepMath-103K/tree/5cf055d1fe3d7a2eb19719ac020211469736ae44) — dataset authors report benchmark decontamination and larger-model RL/SFT results.
- [Pinned MATH-500 evaluation file](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/tree/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be) — public 500-item task set and level metadata.
