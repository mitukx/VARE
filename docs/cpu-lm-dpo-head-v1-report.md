# Cached-model CPU DPO-style update v1

## Question

Does a small, frozen-backbone preference update on an already-cached language model improve preference judgments on disjoint arithmetic-choice prompts, while respecting a fixed drift limit?

The protocol and runner were committed as `657ee63` before formal output was generated. Protocol SHA-256: `df92aba2c0fa90cd6f6c443b376376d1c14b49d50dcd79cffda0e7618c2896d8`. The source repair and first failed attempt were committed as `49cfe26` before the retry. The protocol was not changed after either attempt.

## Method

- Base model: cached `Qwen/Qwen2.5-0.5B-Instruct`, snapshot `7ae557604adf67be50417f59c2c2f167def9a775`; local hash for `model.safetensors`: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
- Runtime: Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.3, NumPy 1.26.4. The runner enforces these versions, offline loading and CPU placement.
- Learner: frozen backbone and base output head, plus a custom rank-4 low-rank residual restricted to the two verified single-token answer-label columns. Full-batch DPO logistic objective, beta 0.1, SGD learning rate 0.05, 15 updates.
- Data: each of three seeds has 24 training comparisons using operands 1–19 and 32 separately generated held-out comparisons using operands 20–39. Each prompt presents the correct sum and a nearby incorrect value in random A/B order. The held-out split is never used for updates or selection.
- Limits: 900 seconds, 6 GiB peak RSS, four CPU threads, no network, GPU or paid service.

This is a custom, narrow output-head intervention. Its metric is preference NLL conditional on choosing between the two answer-label tokens. It is not a general PEFT implementation or a standard TRL run.

## Result

The first execution stopped before an optimizer update because activations were created as PyTorch inference tensors that autograd could not save. The failed attempt, error, runtime, memory, protocol snapshot and runner snapshot remain in [`confirmation/`](../results/cpu-lm-dpo-head-v1/confirmation/). The repaired runner uses ordinary no-gradient activations. The separate `retry-1` run completed in 7.71 seconds with a peak RSS of 3.17 GB.

| Seed | Base held-out NLL | Updated held-out NLL | Change | Accuracy: base → updated | Bernoulli KL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 2.1767 | 2.9406 | +0.7639 | 0.4063 → 0.3750 | 0.0994 |
| 211 | 1.7210 | 1.6023 | −0.1188 | 0.5000 → 0.5000 | 0.6621 |
| 307 | 2.0996 | 1.7904 | −0.3092 | 0.4688 → 0.4063 | 0.4047 |

Mean held-out NLL change (updated minus base) was **+0.1120**. The seed-stratified paired bootstrap 95% interval was **[−0.2673, +0.5002]**. One seed worsened, and one seed exceeded the 0.5-nat per-seed KL ceiling. The frozen decision is **non-pass**. Although the DPO training objective decreased for every seed, held-out preference quality was mixed and the update was not stable enough to pass.

The exact prompts, labels, per-example base/updated margins, adapter parameters, loaded-file hashes, runtime, and manifest are in [`retry-1/`](../results/cpu-lm-dpo-head-v1/retry-1/). The offline auditor reconstructs the generated datasets, per-seed NLL/accuracy/KL, aggregate interval, and decision from these records. It does not rerun model inference or optimization.

## Interpretation and limits

This is the first retained model-level adapter update in VARE, but it does **not** demonstrate a held-out improvement. The current recipe produced unstable results across three initializations. The arithmetic-choice data are small and narrow; the held-out numeric range is a modest task shift, not broad language generalization. The experiment does not support claims about general reasoning, truthfulness, alignment, RLHF quality, deployed behavior, or capability gain.

The next valid iteration needs a separate development-only protocol that selects an update budget using training data alone, followed by a new confirmation lock with fresh seeds and held-out examples. The v1 protocol and non-pass remain immutable. No v2 outcome may be described as confirmation of v1.
