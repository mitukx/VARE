# Qwen2.5 / DeepMath / MATH-500 CPU-first feasibility protocol v2

This revision repairs execution conditions and evidence capture before the first accelerator run. It does not change, reinterpret, or replace the frozen v1 protocol or its retained CPU results. No GPU, MPS, model forward pass, optimizer update, or confirmation output was used while preparing v2.

## Finding: v1's base prompt was not the planned GRPO prompt

The selected `trl-lib/DeepMath-103K` rows have `prompt=[{"role":"user","content": problem}]`. At TRL 1.1.0's GRPO generation path, `_tokenize_prompts` passes the supplied conversation directly to `processing_class.apply_chat_template(..., add_generation_prompt=True, tokenize=True)`. `trl.data_utils.apply_chat_template` adds the generation prompt when the final message is `user` or `tool`. Neither path inserts a system instruction. The v1 base gate added a system message instructing the model to reason and return a boxed answer. Therefore its base success, format rate, and reward variance would not have been evidence for the actual planned trainer input. V2 removes that extra message.

V2 centralizes prompt construction in `scripts/qwen25_deepmath_inputs_v2.py`. Base inference, GRPO rows, SFT prompt-completion rows, and the locked MATH-500 generation contract use one unchanged user message and the model's tokenizer chat template. SFT uses a single assistant completion `\\boxed{verified_answer}` and explicitly sets `completion_only_loss=true`; TRL SFTTrainer's prompt-completion preprocessing applies the same tokenizer template to the user prompt and to prompt plus assistant completion. No custom system prompt or chat-template kwargs are used.

GRPO and base inference are aligned on stochastic decoding: temperature 0.7, top-p 0.95, and 1,024 maximum completion tokens. GRPO's `max_prompt_length` and the base gate's input cap are both 512 tokens. The existing CPU audit measured at most 405 tokens over the fixed 288 selected gate/train prompts, so this bound does not truncate those selected inputs. Held-out MATH-500 uses the identical user-only prompt serialization but predeclared greedy decoding (`do_sample=false`, 1,024 new-token cap) for Pass@1. Sampling differs intentionally between training rollouts and evaluation; prompt content, chat-template source, and generation-prompt handling remain the same. `evaluate_math500_v2.py` scores a completed prediction file; a live MATH-500 generation runner has not been implemented or exercised.

Source inspected: [TRL v1.1.0 GRPO trainer](https://github.com/huggingface/trl/blob/v1.1.0/trl/trainer/grpo_trainer.py), [TRL v1.1.0 prompt chat-template utility](https://github.com/huggingface/trl/blob/v1.1.0/trl/data_utils.py), and [TRL v1.1.0 SFT trainer](https://github.com/huggingface/trl/blob/v1.1.0/trl/trainer/sft_trainer.py). The repository tests assert the locked format and settings. They do not launch TRL or a model.

## Runtime and Colab reproducibility

TRL 1.1.0 declares `requires-python >=3.10`; an exact Python patch pin of 3.11.2 is not a TRL requirement. The GPU runner now accepts Python `>=3.11,<3.13` and still checks every gate library against exact package versions. Google's [Colab past-runtime listing](https://research.google.com/colaboratory/runtime-version-faq.html) includes runtime 2026.07 with Python 3.12.13 and PyTorch 2.11.0, so requiring exactly Python 3.11.2 would reject a documented Colab runtime without establishing a compatibility benefit.

The runner records Python patch version, package versions, CUDA runtime, GPU name, compute capability, and hashes of the protocol, model, data, and selected prompt IDs. Resume requires an exact identity match. A Colab notebook must retain the selected runtime version, install the exact package lock, and record `python --version`, `pip freeze`, `torch.version.cuda`, and `nvidia-smi`. The CUDA wheel and Colab host driver still need to be compatible; that has not been tested because accelerator use is prohibited in this revision.

## Durable gate behavior

`run_qwen25_deepmath_base_gate_v2.py` evaluates only the 32 frozen TRAIN gate prompts, four completions each, and never loads MATH-500 or updates model parameters. The 7,200-second wall budget begins before tokenizer/model loading and is also applied during inference and scoring. On POSIX, a timer raises within Python; a currently running CUDA call can delay signal delivery until control returns. Each finished prompt group is appended and fsynced to a JSONL partial file. Python timeout, OOM, exception, and keyboard interruption leave a JSON state file with the decision and completed-group count. `--resume` skips complete prompt indices, preserves per-prompt seeds, and rejects a changed protocol/model/data/runtime identity. A fault during a prompt can require that one prompt group to be repeated; only earlier completed groups are reusable.

Generated-token accounting sums actual output token IDs, counts the first EOS, and excludes trailing padding. The raw completion IDs and per-output counts remain in each checkpoint record. The global token cap is applied before each group by reducing the allowed token cap per completion; it cannot overshoot the configured total. These properties are tested with CPU fixtures; no CUDA interruption or real OOM has been exercised.

## Independent checker stress tests

The new `math500_grader_v2.py` and tests cover multiple boxed answers, a wrong or malformed final box, empty boxes, equivalent and unequal symbolic expressions, long completions, an overlong final box, parser exceptions, and forced parser/comparison timeouts. The v2 checker bounds full completion length (131,072 characters), final box size (8,192 characters), and main-thread parsing and symbolic comparison to 3 seconds per operation. Over-limit or failed mathematical parses do not receive exact-match credit. Text fallback is limited to explicit text forms.

The suite also mirrors the pinned TRL 1.1.0 `accuracy_reward` extraction/verification calls using `math-verify==0.9.0`. It retains a deliberate disagreement: the independent checker accepts `\\boxed{90^\\circ}` against `\\boxed{\\frac{\\pi}{2}}`, while the pinned reward returns 0. This is why future reports must show training reward and independent task success separately. Reward semantics are checked against the pinned public implementation source; the full TRL package is intentionally not installed in the small CPU checker job.

## CI repair

The 2026-10-10 Actions run failed at collection because the default `pytest` job imported `tests/test_math500_grader_v1.py`, but its minimal project test environment did not install SymPy. The standard suite now excludes the optional mathematical-grader test modules; a dedicated `math500-grader` job installs `requirements/math500-grader-test.lock.txt` and runs v1/v2 checker tests. The rest of the existing workflow remains unchanged.

## Protocol delta from v1

- Preserve every v1 source, lock, result, and conclusion unchanged.
- Remove the base gate's extra system message and tokenize the source user-message array directly with the same template contract used by TRL GRPO.
- Set gate and GRPO prompt cap to the same 512 tokens and make the SFT prompt/completion chat record explicit.
- Remove the exact Python 3.11.2 requirement; keep package versions pinned and record the actual runtime.
- Count tokens from generated IDs, enforce the total token budget, persist prompt-level progress, and permit identity-checked resume after a partial run.
- Harden the independent checker and add CPU adversarial, timeout, cost, and reward-disagreement tests.

The frozen v1 feasibility JSON remains a record of CPU data-shape audits only. It is not relabeled as a model gate result. Only v2 may be used for a future base-policy GPU inference request, and that run still requires a separate user authorization.

## Current decision

**NO-GO for the GPU inference gate today.** Code paths and CPU-only interruption/token/checker behavior are ready for review, but the request explicitly prohibits GPU use. The old v1 gate had prompt mismatch; its CPU package cannot justify a model feasibility claim. V2 is the corrected future protocol, not evidence of model success, memory fit, reward variance, or throughput. The next GPU action, once separately authorized, is one TRAIN-only base inference run with the fixed v2 command and limits. A pass would justify requesting the separate one-update/save-reload resource smoke described by the unchanged study design; it would not authorize multi-seed training.

Command after explicit authorization:

```bash
python -m scripts.run_qwen25_deepmath_base_gate_v2 \
  --model-dir /path/to/Qwen2.5-0.5B-Instruct/snapshot \
  --data-dir artifacts/math-grpo-cpu-first-2026-10-11/data \
  --output artifacts/math-grpo-cpu-first-2026-10-11/gpu-base-gate-v2.json \
  --max-wall-seconds 7200 \
  --max-generated-tokens 131072
```
