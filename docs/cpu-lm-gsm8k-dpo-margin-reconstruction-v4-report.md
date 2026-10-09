# GSM8K DPO adapter-to-margin forensic reconstruction v4

## Question

Do the saved final adapters from the existing GSM8K confirmation, when applied to hidden states recomputed from the pinned cached model, reproduce the recorded base and updated one-token A/B margins and held-out metrics?

This is a forensic reproduction of an already consumed cohort. It performs no training, tuning, row selection, or protocol change for the original confirmation.

## Frozen method

The v4 lock and independent reproducer were committed as `e7d0e5171e77731f9cb33bb39a1b4ee435a1f898` before execution. The lock pins the original confirmation protocol, bundle-manifest hash, model revision/files, seed set, margins and metrics tolerances, CPU budget, and reproducer hash.

The script loads the cached Qwen2.5-0.5B-Instruct model in CPU float32 and recomputes the final prompt hidden state for every retained training and held-out example. It applies each stored rank-4 residual using independent NumPy matrix operations. It imports neither the original runner nor its task helper or offline metric auditor.

For one-token completions `A` and `B`, the same-prompt softmax normalizer cancels:

\[
m_0 = hW_A-hW_B,\qquad m_1=m_0+hA(B_{:,A}-B_{:,B}),
\]

and the DPO logit equals \(y(m_1-m_0)\), where \(y=+1\) when A is preferred and \(-1\) otherwise. The corresponding loss is `softplus(-beta * y * (m1 - m0))`.

The frozen pass criteria required every reconstructed base/updated margin to be within `5e-5`, all train/held-out NLL and accuracy values plus held-out Bernoulli KL, per-seed NLL changes, and bootstrap interval endpoints to be within `5e-5`, and the wall/RSS limits to pass.

## Result

| Check | Reconstructed difference | Frozen limit | Result |
| --- | ---: | ---: | --- |
| Maximum base-margin absolute error | `4.1962e-5` | `5e-5` | Pass |
| Maximum updated-margin absolute error | `4.2021e-5` | `5e-5` | Pass |
| Maximum NLL, accuracy, KL, per-seed change, or bootstrap error | `2.7196e-7` | `5e-5` | Pass |
| Paired bootstrap endpoint error | `1.6439e-9` | `5e-5` | Pass |
| Wall time / peak RSS | `156.78 s` / `3.58 GB` | `3600 s` / `6 GiB` | Pass |

The independently reconstructed held-out NLL changes were `−0.00840346`, `−0.00773650`, and `−0.00477138` nats/question for seeds 401, 503, and 607. Their mean was `−0.006970448`, matching the original result. The reconstructed 95% paired bootstrap interval was `[−0.008003119, −0.005906680]`, matching the retained interval within `1.65e-9`. Mean accuracy remained near chance: `0.4943` at base and `0.4956` after update.

The model and prompt forward pass were recomputed under Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, and NumPy 2.4.4. The original confirmation used Python 3.9.6, PyTorch 2.8.0, and NumPy 1.26.4. The margins therefore pass a tolerance-based cross-runtime check, not a bitwise replay; the maximum margin error is close to the predeclared tolerance.

## Protocol chronology

The v1, v2, and v3 audit locks/scripts are preserved but were not executed. Separate read-only preflight reviews caught, respectively, a protocol/script schema mismatch, incorrect output-path and exception-retention metadata, and a stale protocol path in the script. V4 corrected these before model inference. These are protocol-authoring failures, not experimental non-passes.

## Decision and limitations

**The existing positive small-model result is reproducible through the model-forward → adapter → logit-margin → metric chain.** This strengthens the original same-host metric reconstruction but does not independently rerun the optimizer, establish the correctness of the historical gradient/update implementation, count as outside human reproduction, or provide new confirmation.

The original task remains a public GSM8K forced choice between a verifier answer and a nearby distractor, with a custom adapter restricted to the two answer-label columns. The result does not demonstrate free-form math reasoning, generalization, human preference alignment, or capability gain. Accuracy remains near chance. No GPU, paid API, or network fetch was used; Hugging Face offline flags and `local_files_only` were set, but these are not an operating-system network sandbox.

The next useful step is independent human review or a materially distinct task/update experiment with an untouched exact task-success metric. Do not tune or reconfirm on this consumed cohort.

## Artifacts

- Frozen [v4 protocol](../protocols/cpu_lm_gsm8k_dpo_margin_reconstruction_v4.lock.json) and [reproducer](../scripts/audit_gsm8k_dpo_margin_reconstruction_v4.py).
- [Run bundle](../results/cpu-lm-gsm8k-dpo-margin-reconstruction-v4/run-1/) contains the summary, protocol/script snapshots, command, stdout/stderr, exit code, and hashes.
- Original [GSM8K DPO confirmation report](cpu-lm-gsm8k-dpo-confirmation-v1-report.md) and its input [bundle](../results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/) remain unchanged.
