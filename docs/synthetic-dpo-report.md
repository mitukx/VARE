# Synthetic contextual preference optimization: v1

## Result

Protocol `vare-synthetic-contextual-dpo-cpu-v1` ran locally on CPU for 10 fixed seeds, with no external dependencies or network access. The analytical gradient passed central finite-difference checks. A manifest audit verified all retained inputs, outputs, protocol/lock snapshots, and the exact source snapshot. Raw train/held-out comparisons and seed-level metrics are retained in [`results/synthetic-dpo-cpu-v1-exploratory/`](../results/synthetic-dpo-cpu-v1-exploratory/).

| Condition | Mean held-out preference NLL | Mean held-out accuracy* | Mean KL to uniform |
| --- | ---: | ---: | ---: |
| No update / uniform reference | 0.6931 | Not valid in v1 | 0.0000 |
| Clean-label DPO | 0.3408 | 0.8520 | 0.5629 |
| 20% flipped training labels | 0.4577 | 0.8020 | 0.2056 |
| Shuffled training labels | 0.7002 | 0.5305 | 0.0376 |

Clean DPO reduced held-out NLL by a mean of **0.3523 nats per pair** versus no update. All 10 seeds improved; the paired bootstrap 95% interval was [0.3427, 0.3623]. However, the preregistered acceptance rule also capped mean KL at 0.5, and the observed clean-policy KL was 0.5629. **The protocol acceptance rule failed.** The result does not support a positive acceptance claim.

\* **Accuracy limitation:** v1 counted an exact tie from the uniform reference policy as incorrect, recording reference accuracy as 0.0. For pairwise prediction, a tie should receive half credit (0.5). The v1 reference accuracy and cross-arm accuracy comparison are therefore invalid and must not be cited. A regression test now enforces half credit for ties. NLL and KL values do not depend on this accuracy convention.

## Interpretation

This controlled result shows that the implemented preference objective can move a small synthetic policy toward its known teacher on held-out contexts, but it exceeded the frozen drift ceiling. The clean/noisy/shuffled ordering is descriptive for this generated distribution. It is not evidence of language-model post-training, natural-language preference quality, reasoning, truthfulness, alignment, or real-world capability. The held-out accuracy bug also shows why metric definitions need explicit tie behavior and executable checks.

The next step is a separately versioned protocol and run after correcting the accuracy metric and selecting an update budget using training-only diagnostics. Do not change v1's threshold or reinterpret its acceptance outcome. Keep v1's raw result and limitation visible.
