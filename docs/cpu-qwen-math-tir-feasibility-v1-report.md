# CPU Qwen math tool-use feasibility v1

## Question and frozen hypothesis

Before designing another post-training run, can the already-cached Qwen2.5-Math-1.5B base checkpoint complete a small, independently graded arithmetic task under the declared chat, answer-format, and optional calculator contract, on CPU within a no-cost resource cap?

The frozen screen required all 96 newly generated rows to complete, exact-match accuracy from 25% through 85%, at least 10% accuracy in each of four task families, and compliance with a 6 GiB peak-RSS / two-hour limit. Passing would only justify separately designing an update study. The protocol, data, model revision, thresholds, runner, and auditor were locked before inference. The pilot is excluded from training, development, and confirmation.

## Result

The screen completed all 96 rows in 1,068.2 seconds. Exact boxed-integer match was **0/96 (0%)**, with a two-sided 95% Wilson interval of **[0%, 3.85%]**. Each family scored 0/24. No calculator calls were recorded. Peak RSS was 3,694,985,216 bytes (about 3.44 GiB), and generation took 1,061.1 seconds; both resource limits passed. The frozen advancement gate failed on aggregate and per-family accuracy.

The independent same-host audit regenerated the 96 examples, replayed answer parsing and calculator accounting, recomputed the metrics and decision, and checked the retained bundle hashes. All checks passed. Five frozen objective tests passed under `unittest`. This is a same-host audit, not an external reproduction.

## Interpretation and limits

This retires the **exact model / prompt / decoding / parser / task pairing** under its precommitted rule. It does not establish that the checkpoint has no arithmetic competence or that Qwen models cannot use tools. The primary metric required a final `\\boxed{integer}` expression. For example, event-000 contains the correct value (10,390) in prose but no boxed answer, so the frozen parser marks it incorrect. Many other outputs also fail to follow the requested response contract.

The result therefore demonstrates failure of this format-constrained feasibility screen, not zero latent arithmetic ability. Since no calculator calls occurred and the runner expects a specific textual tool-call wrapper, this run also does not evaluate tool-use capability. The study measures neither an optimizer update nor an independently confirmed capability gain.

## Novelty and decision

This is a feasibility check, not a novel algorithm or a contribution to RL/post-training methods. Its value is a negative gate result retained before any training cost was incurred, plus a reproducible example of why base-checkpoint interaction and output contracts must be validated before opening a learner study.

**Decision: STOP this pairing; PIVOT away from another nearby model/task prompt variant for now.** The repository already contains multiple failed base/task/update attempts and no distinct cached pairing with demonstrated viability. The highest-value next step is independent review or clean reproduction of existing substantive evidence, then a narrowly scoped correctness investigation only if review exposes a concrete defect. Model learning remains gated on a genuinely distinct viable base/task contract and a separate measured update/save/reload path.

## Reproduction

Use the pinned offline environment and already-cached model only:

```sh
python scripts/run_cpu_qwen_math_tir_feasibility_v1.py
python scripts/audit_cpu_qwen_math_tir_feasibility_v1.py results/cpu-qwen-math-tir-feasibility-v1/run-1
python -m unittest tests/test_qwen_math_tir_feasibility_v1.py -v
```

The frozen protocol, lock, generator, pilot, runner, auditor, objective tests, raw records, summary, and manifest are retained in this repository. No network access, paid service, GPU, or model update was used.
