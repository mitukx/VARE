# StrategyQA CPU feasibility gate and verifier-shift novelty decision

**Decision: stop the StrategyQA/Qwen2.5-0.5B-Instruct pairing before optimization.** The frozen base gate produced 0/200 outputs accepted by the required final-answer parser and failed the 80% parse-rate criterion. This is a prompt-following/format feasibility failure; it does **not** establish zero semantic yes/no accuracy. No optimizer step ran, and no training or confirmation labels were used.

## Frozen feasibility screen

Protocol [`cpu_strategyqa_grpo_shift_v2.lock.json`](../protocols/cpu_strategyqa_grpo_shift_v2.lock.json) was committed before inference. The 200-row gate, 200-row confirmation pool, and 32-row train pool are disjoint hash-ranked samples from the official StrategyQA training split. Selection did not use labels. Only the gate questions were generated or scored. The official test file does not include labels; StrategyQA's public train rows were split locally, so model pretraining exposure cannot be excluded.

The run used cached Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, CPU float32, greedy generation, a 32-token cap, and the frozen prompt requesting a concise rationale followed by `Final answer: yes/no`.

| Measure | Result | Frozen gate |
|---|---:|---:|
| Exact parser matches | 0/200 | — |
| Parse rate | 0% | >=80% |
| Accuracy under the required parser | 0% | >=57%, Wilson lower bound >50% |
| Wall time | 396.4 s | <=1,800 s |
| Peak RSS | 3,194 MiB | <=16,384 MiB |
| Paid compute / API | $0 / $0 | $0 / $0 |

The generations frequently began with an unmarked `Yes` or `No` followed by free-form claims, rather than the required final marker. The locked evaluator therefore treated them as invalid. This outcome cannot be repaired by changing the parser or prompt on the same gate cohort. The initial v1 attempt failed before model inference on a runner/lock key mismatch; its failure is retained separately. V2 changes only that key path. The independent standard-library replay passed 13/13 checks, including source and dataset hashes, selection, per-row parsing, metrics, and the failed gate. See the [`v2 run`](../results/cpu-strategyqa-grpo-shift-v2/base-gate/run-1/), [`independent auditor`](../scripts/audit_strategyqa_base_gate_v2.py), and [v1 pre-inference failure](../results/cpu-strategyqa-grpo-shift-v1/base-gate/run-1/failure.json).

## Prior-art and novelty assessment

The proposed broad mechanism—correcting GRPO updates for imperfect verifier labels, then accounting for content-dependent verifier errors that correlate with policy scores—does not survive a novelty gate in its current form.

- Cai et al.'s [October 5, 2026 revision](https://arxiv.org/html/2510.00915v5) derives backward and forward reward corrections, explicitly states the instance-independent/conditional-independence assumptions, gives a covariance residual when errors depend on trajectory features, extends the analysis to group-centered updates, and reports non-i.i.d. noise and correction-misspecification experiments. The broad direction and several natural theoretical extensions are therefore already covered.
- El Mansouri et al. [analyze noise-corrected GRPO under Bernoulli reward flips](https://arxiv.org/abs/2510.18924). More generally, feature-dependent corruption has a substantial instance-dependent label-noise literature.
- VARE's own earlier total-cost analysis already reduced the proposed audit allocation variants to Horvitz–Thompson weighting, cost-aware Neyman/optimal-design allocation, and sequential sign tests. It did not produce a new allocation mechanism.

The remaining gap of measuring whether such corrections improve an independently graded model outcome is scientifically useful, but it is not by itself an algorithmic novelty claim. This StrategyQA screen did not produce usable task outputs; the only retained ARC policy comparison uses a retired model/task/evaluation pairing and a clean exact-answer verifier. Do not imply that VARE validates the proposed verifier-aware intervention.

## Outcome-informed diagnostic of the retained ARC comparison

To understand the prior real-model non-pass without reopening training, a separate descriptive analysis was run against the already retained ARC v3 raw summary. It is explicitly post-hoc and cannot support a confirmatory or causal claim.

- Across 3 seeds there were 96 groups of 4 sampled answers. The exact verifier marked 144/384 responses correct (37.5%).
- Only 50/96 groups (52.1%) contained both a correct and incorrect response. The other 46 groups had zero within-group reward variance: 34 all-wrong and 12 all-correct. Under binary group centering, these groups produce zero relative advantage.
- The all-correct groups contained 48 successful responses, one third of all 144 successful traces. Successful-trace SFT trained on those examples, while the centered GRPO objective gets no within-group preference signal from them. This is a plausible efficiency difference, not proof of why held-out results differed.
- A conditional randomization diagnostic preserved each question's 12 observed outcomes across seeds, then repartitioned them into three groups of four 50,000 times. The observed 50 mixed groups equaled the null mean of 49.62 (SD 2.13; empirical 95% range 45–54; two-sided randomization `p=1.0`). There is no evidence here that seed-defined groups had excess reward correlation after conditioning on prompt-level difficulty.
- Held-out GRPO effects remained mixed: the three seeds changed 9/9, 8/12, and 10/16 items from wrong-to-right/right-to-wrong. The original GRPO-minus-SFT interval remains [−5.22, +2.90] percentage points.

This analysis suggests that binary group-relative signal availability was limited by task-level reward heterogeneity, while ruling out one simple explanation based on extra within-group dependence. It does not identify a correction, demonstrate policy shift, or explain the null held-out outcome causally. Reproduction: `python scripts/analyze_qwen_arc_grpo_group_information_posthoc_v1.py --output results/qwen-arc-grpo-group-information-posthoc-v1/analysis.json --permutations 50000`.

## Research decision

**STOP the current conditional verifier-audit/GRPO correction line as a claimed original contribution.** The newest close prior art covers its broad theoretical mechanism, earlier VARE audit studies found no distinct budget-allocation method, and the fresh CPU model gate failed. Keep the failed screen and diagnostic. Resume model updates only when a materially distinct, no-cost model/task pairing first passes a frozen format and independent-success gate, and only with a narrower hypothesis not already contained in the prior work above.

### Limitations

- The StrategyQA screen is one small instruction model and one frozen prompt; its failure is not a general model-capability result.
- StrategyQA labels come from its official training split because official test labels are unavailable; benchmark pretraining exposure is unknown.
- The ARC group analysis is post-hoc on a consumed cohort; the randomization model is descriptive and treats per-prompt outcomes as exchangeable across seed-defined groups.
- No new policy update, verifier-noise intervention, independent external reproduction, or positive capability result was produced.
