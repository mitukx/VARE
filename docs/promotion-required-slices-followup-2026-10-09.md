# Required evaluation-slice coverage follow-up — 2026-10-09

## Question

Does the current promotion gate reject an evaluation that omits the same required task family from both incumbent and candidate reports?

This is a deterministic source-level correctness diagnostic discovered while reviewing the held-out reward-audit draft. It is not a preregistered research experiment.

## Reproduction

At baseline commit `60d726cf5f4326b1f370a6423d4b785b5c04e21a`, the gate rejected mismatched slice-key sets but accepted this malformed pair because both contained only `easy`:

- incumbent: `{"easy": 0.50}`, primary `0.50`;
- candidate: `{"easy": 0.80}`, primary `0.80`;
- evaluation protocol's required set: `{"easy", "hard"}`.

The full raw baseline output is in [`baseline.json`](../results/promotion-required-slices-v1/baseline.json). The deterministic reproducer is [`reproduce.py`](../results/promotion-required-slices-v1/reproduce.py).

## Change

`PromotionConfig.required_slice_names` optionally declares the protocol-frozen set. When present, both reports must match it exactly. Existing slice-key equality remains enforced when the field is unset. Malformed or duplicate names are rejected when constructing the gate.

With the same inputs, the candidate rejects the omission with `slice_coverage_mismatch`. A positive control containing both declared slices is accepted by both baseline and candidate. Raw candidate output is in [`candidate.json`](../results/promotion-required-slices-v1/candidate.json).

Example:

```python
PromotionConfig(required_slice_names=("easy", "hard"))
```

The caller must freeze this set with the evaluation protocol before measuring or selecting a policy. The code cannot prove that the set itself was not changed after results were seen.

## Validation and limits

- Focused promotion tests: 9 passed.
- Full local suite: 223 passed, 12 skipped, exit code 0. The `PATH` included the VARE virtual environment so nested task commands could find pytest.
- The first full-suite invocation had four unrelated smoke-task failures because subprocesses resolved system Python without pytest. The same failures occurred on the clean baseline worktree; raw records showed `No module named pytest`. With the intended nested-command environment, the suite passed. The full log and exit code are retained in `results/promotion-required-slices-v1/`.
- This is constructed software-contract evidence only. It does not estimate how often real evaluators omit slices, establish evaluator truth or independence, demonstrate a model update, or show a task-success gain.
- The option is opt-in; callers that leave `required_slice_names=()` retain prior behavior.

**Decision: keep the guard as a narrow correctness improvement.** It closes a specified false-accept path while preserving legacy behavior. It is not a new RL mechanism or a capability result.
