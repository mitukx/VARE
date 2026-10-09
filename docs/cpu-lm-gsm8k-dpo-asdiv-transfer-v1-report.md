# Archived GSM8K DPO adapter transfer to ASDiv v1

## Question

Do the three archived GSM8K forced-choice DPO adapters increase preference for the correct numeric answer when answer content is held fixed and the A/B position is reversed, on arithmetic problems outside the GSM8K text cohort?

This is a post hoc transfer evaluation of already-trained adapters. It contains no ASDiv training or model update. The hypothesis was that the previous GSM8K held-out NLL change reflected answer-content sensitivity that survived a dataset change. The frozen success rule required a positive mean change in the counterbalanced semantic margin, a paired 95% interval above zero, and a positive result for each adapter seed.

## Data and protocol

The source is the 2,305-row [EleutherAI ASDiv dataset](https://huggingface.co/datasets/EleutherAI/asdiv), pinned to dataset revision `8f95807222d87b4c688c3c22a6ba2801e1fa03e2`, parquet SHA-256 `79dbad6536fe3e8cd449e67da859d7a6f5ac0d8f852efe563742ca5faee7fb75`. ASDiv is released as CC BY-NC 4.0 and has one public validation split, not a hidden test set. The original [dataset repository](https://github.com/chaochun/nlu-asdiv-dataset) describes its 2,305 English word problems and formula annotations; the [paper](https://arxiv.org/abs/2106.15772) describes its coverage of elementary problem types and language patterns.

No normalized exact overlap was found between either ASDiv question text or `body + question` and any of the 8,792 questions in the cached, pinned GSM8K train/test source. This does not rule out paraphrase, semantic, or pretraining overlap. A Decimal AST interpreter with an explicit arithmetic-operator allowlist independently re-evaluated the equation and scalar answer on 1,550 rows; it rejected 755 rows with unsupported or inconsistent formula/result formats. The evaluator did not execute dataset-provided code. Of those 1,550, 256 rows were selected by a SHA-256 rank fixed before model inference. The source questions are not copied into the repository; users must obtain the pinned public parquet separately.

For every selected problem, the same correct answer and a deterministic distractor exactly one unit away were scored twice: correct answer in A, then correct answer in B. The three saved adapters and frozen Qwen2.5-0.5B-Instruct base were scored on the same prompts. A and B are distinct single-token completions under the pinned tokenizer. No generation or answer parser is involved.

Let `m_A` and `m_B` be the raw `logit(A) - logit(B)` margins when the correct option is respectively A and B. We report:

```text
semantic margin S = (m_A - m_B) / 2
position bias  P = (m_A + m_B) / 2
```

`S` removes a constant A-versus-B preference. The primary metric is the equal-seed-weighted mean of `S_adapter - S_base` over 256 problems. The paired bootstrap resamples problems within each of the three seeds (10,000 resamples, seed 20261010). We also report correct-choice accuracy and NLL averaged over both orientations, Bernoulli KL, and a symmetric Shapley attribution of NLL changes to `S` and `P`.

The paired NLL for one item can be written as:

```text
L(S, P) = [softplus(-(S + P)) + softplus(-(S - P))] / 2
```

The Shapley attribution averages each component's effect when changed first and second, so the content and position terms sum exactly to the observed NLL change. It is a descriptive decomposition, not a causal intervention.

An exact sanity case shows why the distinction matters. At `S = 0`, paired accuracy is 50% for every `P`, while `L(0, P) = log(2 cosh(P/2))`. Reducing a nonzero position bias toward zero therefore lowers paired NLL without increasing semantic margin or paired accuracy. This is an algebraic identity, not a novel result; the experiment tests whether the archived adapters behave this way on this cohort.

The complete frozen protocol and its digest are in [`protocols/asdiv_dpo_semantic_transfer_v1.lock.json`](../protocols/asdiv_dpo_semantic_transfer_v1.lock.json). The run used local CPU, no network during inference, 4 threads, and the cached model revision already used by the GSM8K study.

## Result

The hypothesis failed its preregistered rule. Mean counterbalanced semantic-margin change was **−0.000106 logits**; the paired 95% interval was **[−0.000169, −0.000041]**. All three adapter seeds had negative semantic-margin changes.

| Seed | Semantic margin change | Correct-choice accuracy, base → adapter | Correct-choice NLL change | Position-bias change | Mean KL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 401 | −0.000020 | 49.80% → 50.20% | −0.01253 | −0.08203 | 0.000742 |
| 503 | −0.000146 | 49.80% → 50.20% | −0.01023 | −0.06834 | 0.000525 |
| 607 | −0.000152 | 49.80% → 50.59% | −0.00641 | −0.04179 | 0.000193 |

The equal-seed mean NLL change was **−0.00972 nats**, even though semantic margin declined. The symmetric decomposition assigns **+0.000062 nats** to the content-margin change (slightly worse) and **−0.009785 nats** to the position-bias change (better); their sum reproduces the observed NLL change. On this cohort, the NLL improvement therefore does not support answer-content transfer. It is explained by reduced A/B position bias in the paired evaluation.

An independent full-model-logits replay on 8 deterministically selected items (16 prompts) agreed with the retained base margins exactly and with adapter margins to a maximum absolute error of `1.85e-5`, within the frozen `5e-5` tolerance. The source-formula and cohort audit reconstructed all 256 rows and the primary metric. Inference took 40.93 seconds and peaked at 2.85 GiB RSS.

Raw scores and the auditor result are in [`results/asdiv-dpo-semantic-transfer-v1/run-1/`](../results/asdiv-dpo-semantic-transfer-v1/run-1/). Reproduction requires the frozen Qwen model files, the archived adapter bundle, and the pinned ASDiv parquet at `../work/private/asdiv-dpo-transfer-v1/asdiv.parquet` relative to the VARE checkout. The data file is deliberately not committed under its non-commercial license.

```bash
mkdir -p ../work/private/asdiv-dpo-transfer-v1
curl -L 'https://huggingface.co/datasets/EleutherAI/asdiv/resolve/8f95807222d87b4c688c3c22a6ba2801e1fa03e2/asdiv/validation-00000-of-00001.parquet' \
  -o ../work/private/asdiv-dpo-transfer-v1/asdiv.parquet
shasum -a 256 ../work/private/asdiv-dpo-transfer-v1/asdiv.parquet
python scripts/run_asdiv_dpo_semantic_transfer.py --output results/asdiv-dpo-semantic-transfer-v1/rerun-1
python scripts/audit_asdiv_dpo_semantic_transfer.py results/asdiv-dpo-semantic-transfer-v1/rerun-1
```

## Interpretation and limits

The result establishes a narrow negative: these existing adapters did not improve the counterbalanced correct-answer margin on the selected ASDiv subset. It also shows that lower forced-choice NLL can arise almost entirely from correcting option-position bias, while the semantic-margin metric moves the opposite way. This makes counterbalanced reporting useful when an experiment claims content-sensitive preference learning.

It does **not** establish that the original GSM8K result was caused by position bias; that cohort used one orientation per question and is not reinterpreted here. It does not establish free-form reasoning, downstream task success, an RL method, general transfer, or model capability improvement. ASDiv is a public benchmark and pretraining exposure cannot be excluded. The 256-row estimate is conditional on a formula-supported arithmetic subset, with only three previously trained adapter seeds and one base model. There is no claim of novelty or external reproduction.

**Decision: stop this transfer hypothesis.** Do not retune the adapters or change the test cohort. The next model-learning claim requires a new training protocol whose primary outcome is independently graded answer correctness (or another executable task-success metric), with balanced nuisance factors and held-out tasks. This evaluation is evidence about the previous preference metric's limits, not a substitute for that experiment.
