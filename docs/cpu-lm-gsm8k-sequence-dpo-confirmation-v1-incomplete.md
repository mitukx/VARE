# GSM8K sequence-DPO confirmation v1 — incomplete

The frozen 512-question confirmation did not finish within its 7,200-second CPU wall-time ceiling. At the limit, adapters for seeds 719 and 823 had completed; seed 929 and the three-seed aggregate were incomplete. The run reached **7,205.85 seconds** and **3.70 GB peak RSS** before it was stopped. Its [partial bundle](../results/cpu-lm-gsm8k-sequence-dpo-confirmation-v1/run-1/) retains the protocol, source snapshots, two completed adapter files, failure record, and an audit result marking it incomplete.

No per-seed or aggregate confirmation decision is available. The two adapter files are not an estimate of the protocol's outcome and are excluded from inference. The run is not a pass or a model-quality result.

The 512-question held-out range and 128 update range are considered consumed. A new protocol uses fresh, disjoint train-split rows and a smaller held-out cohort to fit the measured no-cost CPU budget. Its settings and decision rule are fixed before those rows are read.
