# Procedural entailment DPO development v1

## Result

**Frozen development gate: non-pass.** On this synthetic task, contextual DPO did not beat a scalar Yes/No calibration of the frozen model. The base was also below the protocol's minimum quality floor, and the DPO policy exceeded the KL limit. The reserved confirmation cohorts were not generated or evaluated.

| Arm | Development balanced accuracy | Mean conditional KL to base |
| --- | ---: | ---: |
| Frozen base | 0.5078 | 0.0000 |
| Scalar calibration | 0.6211 | 0.0655 |
| Contextual SFT | 0.5573 | 0.4771 |
| Contextual DPO | 0.5573 | 0.6148 |

The base predicted No for nearly every prompt: Yes recall was 0.0156 and No recall was 1.0000. The DPO-minus-base balanced-accuracy difference was +0.0495 (paired, stratified 95% prompt-bootstrap interval [+0.0169, +0.0820]); this missed the +0.05 point-estimate threshold and the base floor failed. DPO was worse than scalar calibration by 0.0638 (interval [−0.1185, −0.0091]); none of the three seeds beat scalar calibration. DPO and contextual SFT tied on mean balanced accuracy, but the DPO-minus-SFT interval [−0.0247, +0.0247] missed the frozen −0.02 non-inferiority bound. DPO's maximum per-seed conditional KL was 0.7420, above the 0.5 limit. These bootstrap intervals are screening statistics: checkpoints were selected on the same development set.

## Protocol and execution

The frozen study compared a CPU-cached Qwen2.5-0.5B-Instruct base against a scalar logit correction, a contextual affine action head trained by answer-key SFT, and the same contextual head trained by binary-action DPO. Training used 128 balanced generated rows; development used 256 balanced rows with a distinct wording style. Each arm used three seeds. The 512-row balanced and 512-row prior-shift confirmation cohorts remain sealed by protocol. The generator assigns Yes/No from a DAG reachability oracle, reconstructs each rendered prompt from its stored graph, and hashes each prompt. Existing pilot rows are excluded by a locked 192-hash denylist.

The run used four CPU threads, offline cached weights, no paid compute, and no accelerator. Wall time was 76.2 seconds and peak RSS was 3.44 GB. The frozen sources were committed before the run at `e43654607952fa410a2dbc94c6f3fd54c591226a`; the one-line fixture correction in `b14c0e9` did not change the runner, generator, or protocol. GitHub CI passed on `b14c0e9` (122 passed, 5 skipped).

The [auditor](../scripts/audit_cpu_procedural_entailment_dpo_development_v1.py) passed. It verified the manifest, exact pre-run Git source snapshots, protocol and pilot-denylist hashes, all 384 generated rows and graph proofs, training-only feature standardization, selected adapters' stored-feature metrics, paired bootstrap intervals, and every gate check. It does not rerun the model forward pass or optimizer; model provenance is the run's recorded local model-file fingerprint.

## Interpretation and limits

This is a falsification of one narrow hypothesis under the locked setup: the contextual binary-action DPO adapter did not add value beyond scalar prior correction and was less conservative under the declared KL measure. The base itself had almost no Yes recall, so the study does not show useful underlying task competence. It is synthetic E2 mechanism evidence only. It does not test human preferences, free-form generation, online RL, real-task success, broad reasoning, or transfer. The failed development gate leaves no basis to spend or open the reserved confirmation cohorts under this protocol.

The [protocol lock](../protocols/cpu_procedural_entailment_dpo_development_v1.lock.json) and [audited bundle](../results/cpu-procedural-entailment-dpo-v1/development/run-1/) retain prompts, graph proofs, hidden features, adapters, metrics, and hashes.
