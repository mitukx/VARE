# HH human-preference DPO development v2

**Decision: non-pass.** A frozen, CPU-only comparison on human-labeled Anthropic HH helpful-base pairs did not improve DPO pairwise preference accuracy over the frozen base policy. The independent same-host score-replay audit passed; the frozen advancement gate did not.

## Question and frozen design

The protocol asked whether a small sequence-level DPO update could improve agreement with held-out HH preferences beyond the frozen policy, while staying within a KL limit and comparing favorably with a matched chosen-only SFT arm. It was frozen in [`cpu_hh_human_dpo_development_v2.lock.json`](../protocols/cpu_hh_human_dpo_development_v2.lock.json) (SHA-256 `6dc90abd997b5bf0247b4a3cd13f93d792e40a2dd36fd2491654e485bf59e777`).

- **Data:** Anthropic HH-RLHF `helpful-base`, pinned dataset revision `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`. The common training cohort has 128 pairs and the hash-disjoint development cohort has 256 unique prompt contexts. Both come from the official training split. The official test archive was hashed for provenance but its rows were not parsed in this study; prior HH experiments have already read that test split.
- **Exclusions:** the lock excludes 1,158 historical context hashes, 64 private feasibility-pilot contexts, and six manually used training rows. The bundle records selected source rows and context hashes, not raw preference text.
- **Model:** cached `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`. Transformer and original output head remain frozen; a custom rank-16 residual is trained over the full vocabulary. This is not a PEFT/TRL implementation.
- **Arms:** sequence-level DPO (`beta=0.1`) and chosen-only SFT, each on the same 128 pairs, adapter, AdamW settings (`lr=1e-4`), two fixed epochs, and seeds 7301, 7311, and 7319.
- **Budget:** offline CPU, four threads, no paid compute, maximum 6 GiB RSS and 7,200 seconds. The completed run used 1,288.32 seconds and peaked at 3,419,340,800 bytes RSS.
- **Primary metric:** raw sequence-sum pairwise accuracy on the 256 held-out development contexts. Advancement requires a base floor of 0.40, a mean DPO gain of at least 0.05, a positive paired prompt-bootstrap lower bound, gains on at least two seeds, and the locked KL and matched-SFT constraints.

## Results

| Metric | Result |
| --- | ---: |
| Frozen base pair accuracy | 0.4141 |
| DPO pair accuracy | 0.4128 |
| DPO minus base | −0.0013 (−0.13 percentage points) |
| Paired prompt-bootstrap 95% interval, DPO minus base | [−0.0117, +0.0104] |
| DPO seed gains over base | +0.0039, −0.0039, −0.0039 |
| Matched chosen-only SFT accuracy | 0.3958 |
| DPO mean pair NLL / base mean pair NLL | 55.8806 / 55.4789 |
| DPO mean token KL to base, by seed | 0.0451, 0.0574, 0.0375 |
| Length-only pair-accuracy baseline | 0.6309 |

The frozen decision is **non-pass**: the required DPO gain and seed consistency failed. Base-floor, KL, NLL-guard, and DPO-versus-SFT non-inferiority checks passed. The DPO-minus-SFT accuracy difference was +0.0169 (95% interval [+0.0026, +0.0352]), but that comparison does not replace the failed primary DPO-versus-base gate. The length-only baseline also substantially exceeds both model policies, so these raw sequence scores should not be read as strong preference prediction.

## Audit and limits

The independent auditor reselected the prompts from the frozen inventory, re-tokenized the rows, recomputed base and retained-adapter prompt scores, and replayed the aggregates, bootstrap interval, and frozen decision. [`audit.json`](../results/cpu-hh-human-preference-dpo-v2/development/run-1/audit.json) reports `status: pass` and the same non-pass decision. This is a same-host offline score replay; it does not independently retrain the adapters, reproduce the run on another machine, or constitute external review.

This is a development result on hash-disjoint contexts from the HH training split, not an official test-set confirmation. The measured outcome is agreement with the dataset's pairwise human labels, not downstream task success or user utility. It is offline DPO, not online RL. The result establishes no general helpfulness, safety, reasoning, capability, or scale claim. No confirmation protocol was opened after the development gate failed.

The retained v1 folder contains protocol, runner, learner, and auditor snapshots but no completed summary or standalone failure record. Accordingly, v1 is not treated here as a decision-bearing result. V2 is the first completed HH human-preference policy-update bundle in the current repository.

## Reproduction

With the pinned model and dataset already cached, run the score-replay audit offline:

```bash
python scripts/audit_cpu_hh_human_dpo_development_v2.py \
  results/cpu-hh-human-preference-dpo-v2/development/run-1
```

The audit checks the retained bundle manifest and frozen protocol before replay. It does not rerun training. The [`protocol`](../protocols/cpu_hh_human_dpo_development_v2.lock.json), [runner](../scripts/run_cpu_hh_human_dpo_development_v2.py), [auditor](../scripts/audit_cpu_hh_human_dpo_development_v2.py), and [bundle](../results/cpu-hh-human-preference-dpo-v2/development/run-1/) preserve the details needed to inspect the run.
