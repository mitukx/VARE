# Qwen2.5-0.5B ARC-Challenge CPU base feasibility v2

**Decision: PASS for the next update-cost smoke only.** This is a base-policy feasibility screen, not evidence that VARE training improves task success.

## Question and frozen protocol

Does the cached Qwen2.5-0.5B-Instruct checkpoint produce a nontrivial exact-answer rate on an unused ARC-Challenge validation sample under a fixed CPU-only protocol, sufficient to justify a task-specific update-cost smoke?

Protocol `qwen_arc_challenge_base_gate_v2` and the runner/auditor source hashes were committed in `a4df5a4` before v2 inference. V2 uses 80 hash-ranked four-choice validation rows, excluding every ID consumed by v1. It pins model revision `7ae557604adf67be50417f59c2c2f167def9a775`, dataset revision `210d026faf9955653af8916fad021475a3f00453`, greedy decoding, the parser, all thresholds, and resource limits. The benchmark test split was not opened. V1's 80 generations are retained as an execution failure because aggregation crashed on a missing machine-readable Wilson z value; v1 scores were not used to change the v2 prompt or thresholds.

Before v2 was frozen, review found that the v2 runner and auditor parsed a small set of answer prefixes differently. They were aligned to the already frozen parser specification, then their final hashes and protocol digest were pinned. Focused grammar fixtures gave identical labels/rejections from both implementations.

## Result

- Exact answers: **36/80 = 45.0%**.
- Wilson 95% interval lower bound: **34.6%**, above the 25% uniform-choice rate.
- Parsed-label rate: **74/80 = 92.5%**.
- Resource use: CPU only, **23.26 s**, peak RSS **2,538 MiB**.
- Frozen gate: **pass**.
- Independent implementation audit: **80/80 records reconstructed**, including selection, prompt hashes, token decoding, parser/scorer, aggregate metrics, and gate; audit status **pass**.

The run bundle is [`run-1`](../results/qwen-arc-challenge-base-gate-v2/run-1/); the independent reconstruction is `independent_audit.json`. The runner and audit use the same host and cached artifacts, so this is not an external reproduction.

## Interpretation and decision

ARC-Challenge and multiple-choice scoring are established; this screen claims no novelty. Public-benchmark pretraining contamination cannot be ruled out. An 80-item base sample does not establish general reasoning ability, post-training benefit, or capability gain.

The result clears only the protocol's next gate: a separately frozen, task-specific CPU update-cost smoke covering a real gradient/update, checkpoint save and reload, and exact-answer behavior after reload on unused validation items. It does not authorize a full learning comparison. Any later comparison must use fresh items, strong matched baselines, multiple seeds, an untouched confirmation cohort, and independent task-success scoring.

## Reproduction

```bash
python scripts/run_qwen_arc_challenge_base_gate_v2.py \
  --output results/qwen-arc-challenge-base-gate-v2/run-1
python scripts/audit_qwen_arc_challenge_base_gate_v2.py \
  --run-dir results/qwen-arc-challenge-base-gate-v2/run-1
```

The exact locked sources, model/dataset revisions, requirements, and output hashes are in the protocol and run bundle.
