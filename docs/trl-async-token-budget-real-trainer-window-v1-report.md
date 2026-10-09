# AsyncGRPO token-budget freshness at the real Trainer step boundary

**Finding:** v3 produced the same ordering on both TRL paths, but independent review found it selected Apple MPS because `use_cpu=True` was omitted. It is a resource-protocol non-pass, not a CPU confirmation. Its trace motivated the corrected v4 run; the successful CPU result is documented separately in the [v4 report](trl-async-token-budget-real-trainer-window-v4-report.md).

This is a reproducible check/use boundary counterexample for the constructed schedule. The configuration's intended freshness boundary is not explicit enough to call it an unambiguous upstream defect: the production stale check occurs at queue yield, before DataLoader prefetch and before the optimizer transition.

## Frozen protocol and implementation

Protocol `trl_async_token_budget_real_trainer_window_v3` and script were frozen in commit `6605c26` before confirmation. They pin TRL `ed8cc2f4337fb9b7b1429db31009ad3577cd5b98` and Transformers 4.57.3, and compare base with the local atomic-admission candidate. A local callback increments policy version at each `on_step_end`, the Trainer boundary where pinned AsyncGRPO registers weight sync. Independent review then found `TrainingArguments` selected MPS on this host. V3 therefore fails its locked CPU resource criterion even though exit codes were zero. Preserve its outputs and do not label them CPU evidence.

The toy objective makes optimizer inputs observable, but it is not a language model, GRPO loss, or capability evaluation. V3 commands, stdout/stderr, parsed results, and environment are retained in [run-1](../results/trl-async-token-budget-real-trainer-window-v1/run-1/). Two earlier frozen harness attempts failed before results because the recorder assumed object samples and then expected `model_version` in the training dict. Their failures are retained under `failed-attempt-v1/` and `failed-attempt-v2/`; v3 maps immutable fixture IDs to preregistered versions because the production queue omits that field.

## Results

| Source | GA | Training inputs by optimizer step | Version at training step | Weights after optimizer steps |
|---|---:|---|---|---|
| Base | 1 | `10 → 11` | `0 → 1` | `0.1000 → 0.0990 → 0.0985` |
| Candidate | 1 | `10 → 11` | `0 → 1` | `0.1000 → 0.0990 → 0.0985` |
| Base | 2 | `10, 11` in one update | `0, 0` | `0.1000 → 0.0990` |
| Candidate | 2 | `10, 11` in one update | `0, 0` | `0.1000 → 0.0990` |

V3 recorded row 11 at current version 0 during collation and version 1 at `training_step` for GA=1. For GA=2, both rows were trained at version 0. However, because v3 used MPS, those values remain a non-pass against the frozen resource constraint; consult v4 for CPU confirmation. Base and candidate traces were identical.

The slight parameter changes are measurements on the deterministic scalar objective, not a causal estimate of policy quality or a comparison against a strict-freshness counterfactual. Different accumulation widths also change optimizer scheduling, so the final weights must not be attributed solely to stale data.

## Reproduction and limitations

The observed v3 environment was Python 3.12, PyTorch 2.9.1, Transformers 4.57.3, and Accelerate 1.12.0 on MPS. The v3 lock did not pin Accelerate's source file. Post-run review recorded installed `accelerate/data_loader.py` SHA256 `8debfb20c6ae58eff3a466cfd67244a12cc8b8d3de00316a6c1d54b0310ee6ca`; v4 subsequently pins it and explicitly selects CPU.

This MPS run does not satisfy the CPU-only protocol. It does not establish prevalence in real AsyncGRPO jobs, realistic GRPO gradient magnitude/direction, downstream task effects, or capability gains. `max_staleness=0` is the strict boundary in the fixture; the pinned config defaults to 4, under which lag 1 is allowed.

**Decision: STOP extending the atomic-admission candidate as a fix.** Retain this independent source-path finding and its negative candidate comparison. No upstream defect claim or PR is justified until the max-staleness use boundary is specified and a correction is shown to preserve update and rollout invariants. The actual optimizer-loop reproduction is an infrastructure correctness finding, not a model-improvement result.
