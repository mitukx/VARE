# Failure triage v1 — synthetic CPU contract

VARE's read-only triage module distinguishes **observable** proxy verifier false-accepts, invalid format, constant-reward GRPO groups, policy/verifier lag, and high token clipping. It returns falsifiable probes, never training instructions or root-cause attributions. This is evidence level E0, not model learning or a validated novel algorithm.

## Reproduce

```sh
python -m pytest tests/test_failure_triage.py
python scripts/reproduce_failure_triage_v1.py --output /tmp/vare-triage.json
vare failure-triage --snapshot /path/to/audit.json --output /tmp/triage-report.json
```

Run the synthetic replay on a fresh checkout. Six deterministic cases cover a clean control, a proxy false accept, output-contract failure, group reward ties, version lag plus clipping, and a deliberately small cohort. The expected outcomes are asserted. Inputs include paired task verdicts, group reward vectors, explicit policy/verifier versions, and SHA-256 digests for train/eval cohorts and proxy/trusted graders.

## Integrity boundaries

- Reject malformed/non-finite evidence, duplicate task/group identities, impossible token counts, future versions, and identical paired dataset/grader hashes.
- Hash inequality is **not** proof of evaluator independence, authentic execution, or disjoint task semantics; a protected independent evaluator is still required.
- Thresholds are heuristic, not calibrated; flags are associations and can coexist. When the sample is insufficient, the system abstains.
- The module does **not** train a policy, mutate RVL replay, promote a checkpoint, or claim capability gain.
- Do not reuse retired VARE confirmation data. A future prospective study needs new preregistered tasks, trusted labels, matched compute, and a truly independent outcome evaluator.

## Next experiment

Export a **new** read-only RVL trace plus separate trusted task labels. Evaluate detection false-positives and ambiguity. If there is reliable separation, freeze an intervention protocol against no-op and simple static controls before measuring independent held-out task success and costs.
