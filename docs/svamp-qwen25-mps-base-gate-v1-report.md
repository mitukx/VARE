# SVAMP/Qwen2.5-0.5B MPS base gate v1

**Decision: fail the frozen feasibility gate; do not train this prompt/model/cohort.** The result primarily exposes a response-format failure. It does not establish low arithmetic competence because semantic correctness on unparseable outputs is unknown.

## Protocol and provenance

The frozen protocol was committed as `ac17225` before any SVAMP model output. It used the cached `Qwen/Qwen2.5-0.5B-Instruct` snapshot `7ae557604adf67be50417f59c2c2f167def9a775`, local MPS, and the pinned MIT-licensed `ChilleD/SVAMP` revision `5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef`. The runner fetched only the 700-row train split (`SHA-256 bb3b4cd2…76675e2b`); the 300-row official test split was not opened.

An independent Decimal AST interpreter evaluated the released `Equation` fields. It found one train-row inconsistency (`chal-680`: equation evaluates to 5, stored answer is 1); this row was excluded under the frozen rule. The remaining 699 rows produced 599 training-only and 100 development rows. Exact problem-text comparison against the 1,319 consumed GSM8K confirmation questions found no normalized exact overlap. This does not rule out semantic overlap or pretraining contamination.

The gate scored 64 fresh development rows greedily and sampled four responses each for 16 of those rows. The parser required exactly one `<answer>number</answer>` tag and exact Decimal equality. The locked pass criteria were at least 13/64 exact, 61/64 parseable, 4/16 mixed-reward groups, under 90 minutes, and under 12 GiB peak RSS.

## Results

| Measure | Result | Gate |
|---|---:|---:|
| Greedy parseable | 3/64 (4.69%; Wilson 95% CI 1.61–12.90%) | ≥61/64 |
| Greedy exact, counting parse failures incorrect | 3/64 (4.69%; same interval) | ≥13/64 |
| Exact among parseable outputs | 3/3 | diagnostic only |
| Sampled parseable | 1/64 | reported, not gated separately |
| Mixed-reward groups | 0/16 | ≥4/16 |
| Wall time | 545.31 s | ≤5,400 s |
| Peak RSS | 2.40 GiB | ≤12 GiB |

Most greedy completions did not emit the required answer tag; inspection showed prose, LaTeX/markdown, or truncated completions. The 61 unparseable answers were not post-hoc regraded, so their semantic correctness remains unknown. Only three greedy outputs were scorable under the frozen contract, and all three were correct. The sampled group arm yielded only one scorable response and no correct rewards; all 16 groups therefore had constant observed reward. The exact-reward GRPO signal was absent under this prompt/format setup.

## Interpretation and next gate

The compute gate passes for short-context MPS inference, but the task/reward gate fails. This is not evidence that Qwen has 4.7% underlying SVAMP accuracy; it is exact success under a deliberately frozen response contract. No policy update, held-out model-improvement claim, or official-test access occurred. Preserve this non-pass and retire this exact prompt/model/cohort; do not retune against these 64 rows.

A single targeted follow-up may test whether the failure was format compliance rather than arithmetic: use a simpler numeric-only prompt and a new hash-selected cohort drawn from the previously untouched portion of the SVAMP train split. Freeze its parser and gate before inference. The official test split remains unopened. If that fresh base gate fails, stop this pairing instead of adding prompt variants.

Reproduce with:

```bash
python3 reproducers/run_svamp_qwen25_05b_mps_base_gate_v1.py
```

Raw outputs and metrics are in [`run-1`](../results/svamp-qwen25-05b-mps-base-gate-v1/run-1/); the protocol and runner are [`locked here`](../protocols/svamp_qwen25_05b_mps_base_gate_v1.lock.json) and [`implemented here`](../reproducers/run_svamp_qwen25_05b_mps_base_gate_v1.py).
