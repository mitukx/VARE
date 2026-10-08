# Synthetic preference noise and shift study v1

## Question

Under a fixed update and data budget, how does a small contextual preference policy respond to flipped training labels and to a changed preference function at evaluation? This is a controlled mechanism study. It does not test a language model, natural-language preferences, reasoning, truthfulness, alignment, or real-world capability.

## Frozen design and audit

The protocol and implementation were committed at `526c68e0a10a4bb80e0f777726171b92d4cd2306` before the formal run. The lock fixes ten seeds, a four-action/eight-feature linear softmax policy, 64 training contexts with eight comparisons each, 100 full-batch updates, and 128 disjoint held-out contexts with two comparisons each. The learner is evaluated on matched held-out context/action-pair designs with labels from both the base teacher and a shifted teacher.

Training arms use the original preferences, or flip each pair's orientation at fixed 20% or 40% thresholds. Shared per-example random draws make the 20% flip set a subset of the 40% set, and a flip always preserves the original pair. The primary decision is clean-arm held-out base-teacher NLL improvement over a frozen uniform policy, subject to a paired bootstrap lower bound, a minimum seed count, and a KL ceiling. The noisy and shifted results are descriptive; they do not enter acceptance.

The runner uses only Python's standard library and local CPU. It retains all designs, labels, per-seed metrics, exact source snapshots, environment fields, and file hashes in the [confirmation bundle](../results/synthetic-preference-robustness-v1/confirmation/). Run the [auditor](../scripts/audit_synthetic_preference_robustness.py) to regenerate all data and reconstruct 80 arm-by-condition-by-seed records and the primary decision. The retained [audit output](../results/synthetic-preference-robustness-v1/audit.json) confirms disjoint training/held-out contexts and paired base/shift evaluation designs. The formal run took **7.12 seconds** on the recorded local machine; peak RSS was **22,790,144 bytes** on macOS as reported by `getrusage`.

## Result

The frozen primary rule passed. Mean clean-arm held-out base-teacher NLL improvement over the uniform reference was **0.3174 nats per pair** (paired seed bootstrap 95% interval **[0.3021, 0.3328]**); all **10/10** seeds improved. Mean policy KL to uniform was **0.3952**, below the locked ceiling of **0.5**. Mean clean-arm base-teacher held-out accuracy was **0.8398**, versus **0.5000** for the tied uniform reference.

The label-flip arms degraded on every seed relative to clean training. Mean base-teacher NLL was **0.4798** at 20% flips and **0.6282** at 40%, compared with **0.3758** for clean training. Their mean KL values fell to **0.1541** and **0.0487**, respectively. This pattern is consistent with increasingly contradictory pair labels shrinking the learned policy movement in this specific setup; it is not a general estimate of robustness to human annotation noise.

Under the shifted teacher, mean held-out NLL was **0.4551** for the clean-trained policy, versus **0.3758** under the base teacher. The changed preference function therefore makes this policy's fit worse on average in this fixture. The 20% and 40% flip arms had mean shifted NLL **0.5173** and **0.6420**. These values are descriptive because the protocol's sole acceptance rule concerns clean training and the base teacher.

## Interpretation and limits

This confirms that the frozen objective, paired data construction, independent held-out measurement, and audit behave as intended on a known synthetic preference problem. It also records a controlled sensitivity to label flips and one explicit preference shift. The teacher, policy class, labels, and data distribution are all synthetic and small. The study does not establish preference learning from language, model behavior improvement, reward-model quality, RLHF/DPO effectiveness on user data, transfer, or production impact. The earlier cached-model update remains a non-pass; this experiment does not change that result.

## Reproduction

From the repository root, run:

```bash
python3 scripts/run_synthetic_preference_robustness.py \
  --output results/synthetic-preference-robustness-v1/confirmation
python3 scripts/audit_synthetic_preference_robustness.py \
  results/synthetic-preference-robustness-v1/confirmation
```

The runner refuses to overwrite an existing output directory. No GPU, download, package installation, paid API, or external compute is required.
