# Qwen3-4B DeepMath GRPO/SFT study — protocol v1

**Status as of 2026-10-13: protocol frozen; model inference and optimizer updates have not started.** The machine-readable [protocol lock](../protocols/qwen3_deepmath_grpo_math500_v1.lock.json) pins the model and dataset revisions and hashes, fresh training-only cohort fingerprints, runtime versions, execution-source hashes, outcomes, and compute caps. CPU data/cohort audit and the local unit tests pass. Colab reports an allocated A100-SXM4-40GB. The pinned model and data have not yet been downloaded to that runtime.

## Research question and claim boundary

On a small math task, does four-completion online group-relative correctness optimization improve independent MATH-500 exact-answer Pass@1 over both the unchanged policy and answer-only SFT trained on the same verified examples?

This protocol compares one model on one public benchmark with five paired training seeds and a fixed answer checker. It cannot establish general reasoning gains, human-preference quality, semantic decontamination, or recursive self-improvement.

## Previous feasibility result

The saved Qwen2.5-0.5B train-only gate record reports 10/128 exact successes, 101/128 valid final boxes, and 9/32 mixed-reward groups; it missed that gate's final-box threshold and stopped before an update or MATH-500 scoring. Its raw outputs have not been independently regraded in this work. Its 288 selected prompt IDs are excluded from the new cohort.

## Frozen inputs and execution

- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0; all expected file hashes are in the lock.
- Training data: pinned `trl-lib/DeepMath-103K` TRAIN parquet, 97,870 raw rows. The audit removes duplicate excess rows and excludes conflicting or unparseable groups. New gate (32) and update (256) IDs use independent salted rankings and exclude the retired 288 IDs; both selected cohorts are disjoint and their digests are locked.
- Evaluation: pinned MATH-500 revision. The 100-item development IDs and 400-item confirmation IDs are disjoint and locked. Confirmation scoring stays sealed unless the development advancement rules pass.
- Runtime: observed Colab Python 3.13.15, PyTorch 2.11.0+cu130 / CUDA 13.0, TRL 1.1.0, Transformers 5.18.0, and exact supporting package pins. The execution runners verify runtime, GPU, model/data, protocol, and source hashes. Runs and artifacts are stored outside the Git checkout.
- Prompt: one system instruction and one user problem through the pinned model tokenizer template: `Please reason step by step and put your final answer within \\boxed{}.` No prompt or threshold adjustment is allowed after the base gate begins.

## Phases and stop rules

### 1. Fresh base-policy signal gate

Sample four completions for each of 32 new TRAIN-only prompts using temperature 0.7, top-p 0.8, top-k 20, and a 2,048-token completion cap. Record prompt IDs, token IDs, reward and independent exact-match results, valid-box results, group variance, and resource use. Continue only if all locked criteria pass: at least 10 mixed-reward groups, positive mean within-group reward variance, no reward/checker disagreements, at least 95% valid boxes, task success between 5% and 90% inclusive, under four hours and 262,144 generated token IDs, and below 90% reserved VRAM. A failure retires this exact gate and ends the study before an optimizer update.

### 2. One-update GRPO integrity smoke

Only after a passing gate, run one optimizer update using TRAIN rows. Verify that trainable LoRA weights change and remain finite, frozen base weights remain identical, rollout groups and reward provenance are complete, and the adapter plus optimizer state survive exact save/reload checks. This phase does not load or score MATH-500. The smoke has a one-hour wall limit and the same 90% VRAM ceiling.

### 3. Matched training and development decision

For each of five locked seeds, train answer-only SFT and GRPO from the same base on the same 256 prompts, with 64 optimizer updates per arm. Both use BF16, LoRA rank 16 / alpha 32 over q/k/v/o, AdamW, a linear schedule, and gradient checkpointing. SFT uses completion-only loss over verified boxed answer targets, per-device batch 1 and accumulation 4. GRPO uses the stock TRL accuracy reward, group-relative scaling, four generations, per-device batch 4 and accumulation 4. Inspection of the pinned TRL sampler and a live fixture confirms four distinct prompts with four completions each per optimizer update. The comparison matches unique prompt IDs, base model, updates, and paired seeds; it does not match total generated FLOPs, completion tokens, or SFT supervised tokens.

After all ten adapters pass file-integrity checks, greedily score the unchanged base and all adapters on the fixed 100-item development split. Advance only if mean GRPO exact match is no lower than base or SFT, at least three of five paired GRPO-minus-SFT seed differences are nonnegative, and every GRPO seed has at least 95% valid final boxes. Otherwise stop with confirmation unopened.

### 4. Confirmation, only after advancement

Greedily score base and all ten adapters on the reserved 400-item split. Report task-level and per-seed results, per-level results, format validity, generated-token counts, wall time, and peak GPU memory. Compute paired task-only, seed-only, and hierarchical seed-then-task bootstrap intervals. A positive confirmation requires mean GRPO-minus-SFT of at least two percentage points, a hierarchical 95% interval lower bound above zero, at least four of five positive paired seed differences, no level regression worse than five percentage points, and at least 95% valid boxes for each GRPO seed. The rule is conjunctive; all other outcomes remain non-passes.

## Evidence and limitations

Local tests cover prompt/cohort construction and paired bootstrap behavior. The CPU audit verified the pinned dataset hashes, 32/256 new cohort fingerprints, 288 retired IDs, answer targets, and MATH-500 split digests. A live Colab smoke verified the parser, stock reward, package versions, and exact RepeatSampler batch construction. These checks do not constitute model inference, training, or evaluation.

The training reward and evaluation use separate exact-match paths but share a low-level LaTeX parsing dependency. Exact normalized prompt overlap is zero; semantic near-duplicates and model pretraining exposure are not ruled out. MATH-500 is public, and its reserved confirmation subset is protocol-reserved rather than secret.

No claim about the experiment outcome is made until the relevant phase has actually run. Do not alter this lock in response to observed model scores; any changed design requires a new version and unused cohorts.
