# Reading the BoolQ v17 post-training study

This note follows one completed study from its hypothesis through its frozen decision. It is an audit guide; it does not replace the [full report](boolq-posttraining-development-v17-report.md), protocol locks, or retained run bundles.

## Question and setup

The preceding v16 comparison showed that DPO could lower verifier-preference loss while reducing task accuracy, and that the SFT and anchored-DPO checkpoints exceeded the KL limit. V17 tested a tenfold lower learning rate to see whether matched updates could remain close to the base policy without hurting passage-grounded Yes/No accuracy.

All arms used the same cached Qwen2.5-0.5B-Instruct snapshot, rank-16 output-head residual, 128 training pairs, 256 validation questions, and three seeds. Training and validation rows were chosen by deterministic SHA-256 ranks within separate official dataset splits. V17 used train ranks 152–279 and validation ranks 768–1023. The reserved confirmation ranks 1024–1535 were not opened.

The transformer and original language-model head stay frozen. For hidden state `h`, the custom adapter changes the logits as:

```text
logits(h) = h W_lm + (h A) B
A ∈ R^(896×16), B ∈ R^(16×vocab_size)
```

`B` starts at zero, so the adapter initially preserves the base logits. Only `A` and `B` are trained. This is a custom output-head residual, not a full-model update or a PEFT/TRL run.

## Matched learning objectives

Each training pair has a verifier-chosen completion `y+` and a rejected completion `y−`. The chosen answer comes from the BoolQ answer key; the reject is an actual base-model rollout. For DPO, the implementation sums completion token log-probabilities, subtracts the frozen reference margin, and applies:

```text
mπ = log π(y+|x) − log π(y−|x)
mref = log πref(y+|x) − log πref(y−|x)
loss_DPO = softplus(−0.1 × (mπ − mref))
```

The three arms were:

- **DPO:** the pairwise loss above.
- **Answer SFT:** token-mean cross-entropy on chosen completions.
- **Anchored DPO:** DPO loss plus `0.05 ×` token-mean chosen-completion NLL.

The same data, initialization seeds, rank, optimizer, learning rate, and checkpoint schedule make the objective the planned difference among arms. The preference labels remain answer-key/verifier labels, not human annotations.

## Selection and advancement rule

For binary labels, balanced accuracy is the mean of the Yes and No class accuracies:

```text
balanced_accuracy = (accuracy_Yes + accuracy_No) / 2
```

This prevents the larger class from dominating the score. Checkpoints first had to improve verifier-preference NLL over base and stay at mean token KL ≤0.5. Among eligible checkpoints, the study selected the highest mean validation balanced accuracy. To advance, an arm also needed at least 128/256 base exact matches, base balanced accuracy ≥0.50, mean gain ≥0.05 (five percentage points), and at least two of three seeds no worse than base.

Checkpoint selection used the same 256 questions as the reported comparison. The paired bootstrap interval is therefore descriptive; only the separately reserved cohort could provide confirmation.

## Observed result

| Method | Exact match | Balanced accuracy | Mean token KL | Decision |
| --- | ---: | ---: | ---: | --- |
| Base | 63.67% | 66.84% | 0 | Reference |
| DPO | 63.67% | 66.84% | 0.000113 | Non-pass |
| Answer SFT | 63.80% | 66.94% | 0.000400 | Non-pass |
| Anchored DPO | 63.67% | 66.84% | 0.000258 | Non-pass |

All selected arms lowered verifier-preference NLL and stayed under the KL cap. DPO and anchored DPO retained the base model's task predictions. SFT gained one correct answer in one seed and matched base in the other two; its mean balanced-accuracy gain was 0.11 percentage points, far below the five-point threshold. No method advanced.

## What the audits establish

Each of the three arm auditors reconstructed its locked rows, checked all 15 adapter checkpoints, reproduced checkpoint selection and the non-pass decision, and compared three retained generations with the pinned Hugging Face generation path. The paired comparator checked that all arms used the same cohorts, seeds, and base generations before calculating their differences.

These audits support consistency of the retained bundles under the audited code. They do not show independent human reproduction, human-preference quality, or that the labels predict user preferences.

To inspect the record, start with the [study plan](boolq-posttraining-study.md), then the [full report](boolq-posttraining-development-v17-report.md), [paired comparison](../results/cpu-lm-boolq-posttraining-development-v17-comparison.json), three run bundles under `results/cpu-lm-boolq-posttraining-development-v17-*/run-1/`, and the [current evidence gaps](current-gaps.md). Per-arm locks are in `protocols/cpu_lm_boolq_posttraining_development_v17_*.lock.json`.

## Limits

This is a development-only result on one small public benchmark and one cached model. BoolQ may have appeared in pretraining. The preference pairs are generated from answer keys and base-model outputs; they are not human preferences. The experiment found no meaningful held-out task gain, opened no confirmation data, and does not establish broad reasoning, truthfulness, general preference alignment, or capability improvement.
