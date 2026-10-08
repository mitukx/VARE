# Synthetic preference noise and shift study v1

## Question

Under a fixed update and data budget, how does a small contextual preference policy respond to flipped training labels and to a changed preference function at evaluation? This is a controlled mechanism study. It does not test a language model, natural-language preferences, reasoning, truthfulness, alignment, or real-world capability.

## Frozen design and audit

The protocol and implementation were committed at `526c68e0a10a4bb80e0f777726171b92d4cd2306` before the formal run. The lock fixes ten seeds, a four-action/eight-feature linear softmax policy, 64 training contexts with eight comparisons each, 100 full-batch updates, and 128 disjoint held-out contexts with two comparisons each. The learner is evaluated on matched held-out context/action-pair designs with labels from both the base teacher and a shifted teacher.

Training arms use the original preferences, or flip each pair's orientation at fixed 20% or 40% thresholds. Shared per-example random draws make the 20% flip set a subset of the 40% set, and a flip always preserves the original pair. The primary decision is clean-arm held-out base-teacher NLL improvement over a frozen uniform policy, subject to a paired bootstrap lower bound, a minimum seed count, and a KL ceiling. The noisy and shifted results are descriptive; they do not enter acceptance.

The runner uses only Python's standard library and local CPU. It retains all designs, labels, per-seed metrics, exact source snapshots, environment fields, and file hashes in the [confirmation bundle](../results/synthetic-preference-robustness-v1/confirmation/). The original [auditor](../scripts/audit_synthetic_preference_robustness.py) regenerates designs/labels and reconstructs 80 arm-by-condition-by-seed records and the primary decision, but imports the runner's data-generation, training, evaluation, and bootstrap functions. Its pass therefore does not rule out a defect shared with those functions. The retained [audit output](../results/synthetic-preference-robustness-v1/audit.json) confirms disjoint training/held-out contexts and paired base/shift evaluation designs.

A second [independent bundle verifier](../scripts/verify_synthetic_preference_robustness_bundle_independent.py) imports neither implementation. It replays teacher labels from retained action-pair designs, reconstructs flipped labels, re-trains all 30 non-reference policies, recomputes all 80 arm/condition/seed metric records and the locked decision, and verifies the full artifact manifest. It passed with mean clean-arm NLL improvement 0.3174 (95% interval [0.3021,0.3328]), all 10 seeds improved, and mean KL 0.3952. A mutation test confirms it rejects altered metrics even after the manifest is rebuilt. The replay produced identical metrics using Python 3.9.6, 3.11.15, and 3.12.12 on the same macOS machine; see the three [retained independent audit records](../results/synthetic-preference-robustness-v1/independent-audits/). This replay does **not** regenerate Gaussian contexts or sampled action-pair designs; the protocol's use of `random.Random.gauss`/`sample` does not freeze their behavior across Python versions. No outside person has reproduced the study. The original formal run took **7.12 seconds** on the recorded local machine; peak RSS was **22,790,144 bytes** on macOS as reported by `getrusage`.

After public commit `70e4660` was pushed, a fresh depth-1 clone of GitHub `main` reproduced the same verifier result under Python 3.12.12. It verified all 16 bundle files, replayed the training/metrics, and matched the frozen acceptance decision. The [replay record](../results/synthetic-preference-robustness-v1/independent-audits/github-clean-clone-python-3.12.json) and [clone provenance](../results/synthetic-preference-robustness-v1/independent-audits/github-clean-clone-provenance.json) pin the result to that commit. This is a clean-clone run by the project author, not outside-person reproduction; it also leaves design-generation replay unresolved.

Reconstruction also passed from a detached clean clone of commit `8549561`. The repository's Ubuntu/Python 3.11 [CPU evidence workflow](https://github.com/mitukx/VARE/actions/runs/37717689087) reran and audited the study successfully. The retained run used Python 3.9.6 on macOS, while CI used Python 3.11; because `gauss` and `sample` behavior is not pinned by this protocol, CI is not bit-for-bit reproduction of the retained designs. The clean clone was run by the project author; the CI job is independent execution infrastructure, not independent human review.

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
python3 scripts/verify_synthetic_preference_robustness_bundle_independent.py \
  results/synthetic-preference-robustness-v1/confirmation
```

The runner refuses to overwrite an existing output directory. No GPU, download, package installation, paid API, or external compute is required.
