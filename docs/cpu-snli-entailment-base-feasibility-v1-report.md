# SNLI entailment base-policy feasibility v1

## Decision

**Non-pass; retire this task formulation and cohort.** Before any parameter update, the pinned Qwen2.5-0.5B-Instruct policy scored 49.41% balanced accuracy on the locked 512-row balanced binary screen. Its 95% stratified bootstrap interval was [45.31%, 53.71%]. The frozen gate required balanced accuracy of at least 58%, both class recalls of at least 40%, and a bootstrap lower bound above 50%. The balanced-accuracy and interval checks failed. No model update or confirmation followed.

The result rules out proceeding with this exact SNLI task/prompt/model setup under the preregistered decision rule. It does not show that the model is generally incapable of entailment, nor does it estimate performance outside this selected sample.

## Frozen task and data

The study converted Stanford SNLI into a balanced binary choice: entailment versus neutral-or-contradiction. It used 256 rows per class from the pinned official validation split. Within each binary class, candidates were ranked by the SHA-256 of premise plus newline plus hypothesis and original row index. Rows sharing a premise with the prior format pilots, or matching their typed prompt-pair hashes, were excluded before ranking. The study did not load the official test split.

For every selected row, the prompt displayed both meanings in a deterministic, hash-keyed random order. The model was scored only on the next-token logits for the single-token completions ` A` and ` B`; it did not generate a response. The decision is therefore a base-policy forced-choice screen, not a free-form answer evaluation.

The pre-run protocol is [`snli_entailment_base_feasibility_v1`](../protocols/snli_entailment_base_feasibility_v1.lock.json), SHA-256 `5735b81ea07267d00db9b99555dcb1ab65713529601044d7545f04909541cbd7`. The source was frozen at commit `be10384e3c8ddb2d0902dd06b3da9ccd41d8524e`; its GitHub CI run [37824623642](https://github.com/mitukx/VARE/actions/runs/37824623642) passed before the model screen.

## Results

| Measure | Result | Frozen rule | Outcome |
| --- | ---: | ---: | --- |
| Balanced accuracy | 0.4941 | ≥ 0.58 | Fail |
| Entailment recall | 0.4961 | ≥ 0.40 | Pass |
| Not-entailment recall | 0.4922 | ≥ 0.40 | Pass |
| Stratified bootstrap 95% interval | [0.4531, 0.5371] | Lower bound > 0.50 | Fail |
| Confusion matrix, true rows / predicted columns | `[[127,129],[130,126]]` | — | — |

Accuracy equals balanced accuracy here because the selected cohort contains 256 examples per class. Both recalls are near chance; their individual threshold passes do not offset the failed primary and interval gates.

## Execution and audit

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; cached model-file hashes are retained in the bundle.
- Data: `stanfordnlp/snli`, revision `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`; validation fingerprint `6c5035fc008e4d4e`; Arrow SHA-256 `a688c8f06b3a28f6c9e52e9eb919e576c975e4161ba0139e5aafc6704345cd38`.
- Runtime: Python 3.9.6, PyTorch 2.8.0, Transformers 4.57.3, Datasets 4.4.2, NumPy 1.26.4, Tokenizers 0.22.1, Safetensors 0.7.0.
- Study: CPU only, four threads, offline/local files, zero updates, no paid compute; 76.60 seconds and 3,873,832,960 bytes peak RSS.
- Independent same-host audit: passed; reconstructed the ranked cohort and independently replayed all 512 model scores, metrics, resource record, and gate. Audit took 73.79 seconds and peaked at 3,874,144,256 bytes RSS.

The audit is a separate implementation on the same host and runtime. It is not an outside-person reproduction. The complete selected prompts, row hashes, logits, fingerprints, source snapshots, audit, and 15-file SHA-256 manifest are in [`run-1`](../results/snli-entailment-base-feasibility-v1/run-1/).

## Limits and next action

This screen has no training or policy update, uses a small public model that may have seen SNLI during pretraining, and has no separate confirmation sample. Its output only tests whether this exact base policy clears the predeclared feasibility screen on the selected validation rows. Because it failed, the cohort and task formulation are retired: do not lower the threshold, tune prompts against these scores, train on these rows, or reuse them for another claim. The next model experiment needs a different task or feedback signal, a fresh cohort, a viable base rate, and its own frozen gate before scoring.
