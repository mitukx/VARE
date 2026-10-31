# Offline G1/G3 verifier-error pilot on run-001 train rollouts: aborted (negative result)

Offline replay of the run-001 train rollout log `../rollouts/graded.jsonl` (not committed; sha256
`cb18ed4af65a4f542008a49b6a42d8c8ee3b848cf0775ea292e47f63a9e121b3`, listed in `../SHA256SUMS.txt`).
No model, no new inference, no test execution, no held-out rows (the script rejects any row whose
`split` is not `train` or whose `stage` is not `rollout`). Linux, Python 3 + numpy.

## Verdict

**Aborted by its own pre-registered sample-size rule: 7 informative groups < 8.** G1 and G3 are both
`inconclusive`. This is a negative result about the *precondition*: on this task distribution the
F2P-only cheap verifier made no errors, so budgeted correction of its errors cannot be tested here.
Nothing is claimed for or against G1/G3.

## What was pre-registered (hypotheses, not findings)

- **G1**: with a trusted-audit budget of 10% of rollouts, a Horvitz–Thompson residual correction
  `r = c + a (t - c) / pi` of the cheap reward gives group-relative advantages better aligned with the
  trusted-label advantages than the cheap reward alone.
- **G3**: allocating audits in proportion to (advantage sensitivity × posterior error uncertainty / trusted
  cost) beats uniform audits at the same budget.
- Methods: B2 cheap-only, B3 audit-only, B4 uniform HT, B5 uniform audits with backward rate correction,
  B6 adaptive HT, B6 without the sensitivity term (ablation). Cheap labels: the logged `cheap_f2p_v1` run
  and synthetic flips at (rho0, rho1) = (0.1, 0.1), (0.3, 0.1), (0.1, 0.3). 200 audit/flip seeds,
  2,000-replicate bootstrap over task groups × seeds.
- Primary metric: `E_seed[cos(A_hat, A*)]` over all rollouts (linear-proxy policy-gradient alignment;
  invariant to positive rescaling, 0 for an all-zero estimate). Thresholds are proposed values, not derived
  from data. Full rule set: the `PREREG` dict in `pilot_analysis.py`.
- Abort rule: fewer than 8 informative groups (trusted labels or cheap labels mixed within a group) ⇒
  `inconclusive`, no further analysis.
- Pre-registration hash (sha256 of `json.dumps(PREREG, sort_keys=True)`):
  `be5579fde3971c04da56cf9db033e95c381698f5c1f62e0e33c3953a8f10e887` (v2).
- v2 superseded v1 (`90dd5a9e05b15880d1a2744e9956e6569df795a329924723d8b985ec47069bdd`) **before the
  real log was opened**. v1 used advantage MSE as primary; on synthetic data MSE was minimised by shrinking
  advantages toward 0 (audit-only won by zeroing most advantages), so it was replaced by the alignment metric.
  The change and its reason are also recorded inside `PREREG` (`supersedes`, `change_note`).

## Verified results (recomputed; `pilot_result.json` is byte-identical on rerun)

| quantity | value |
|---|---|
| rollouts / groups (G=4, base checkpoint, step 0) | 160 / 40 |
| repositories in the log | 3 |
| trusted pass (`trusted_full_v1`) | 32 |
| cheap pass (`cheap_f2p_v1`, separate F2P-only run) | 32 |
| cheap false positives / false negatives | 0 / 0 |
| rho0 = P(cheap pass \| trusted fail), Wilson 95% | 0 / 128, [0, 0.0291] |
| rho1 = P(cheap fail \| trusted pass), Wilson 95% | 0 / 32, [0, 0.1072] |
| informative groups (minimum 8) | 7 → abort |
| label consistency: reward vs trusted tests, cheap `passed` vs cheap tests, direct vs derived F2P label | 0 / 0 / 0 mismatches |

Descriptive breakdown from the same log (not pre-registered):

| repository | tasks | trusted pass / samples | mixed groups | all-pass groups |
|---|---:|---:|---:|---:|
| Suor/funcy | 16 | 6 / 64 | 4 | 0 |
| andialbrecht/sqlparse | 16 | 22 / 64 | 3 | 4 |
| gweis/isodate | 8 | 4 / 32 | 0 | 1 |

Outcomes: 32 pass, 91 applied but wrong, 37 malformed/unapplied. In all 91 applied-but-wrong edits at
least one FAIL_TO_PASS test failed, which is why the F2P-only verifier never accepted a wrong edit.
Median verifier wall time on applied edits: cheap 0.25 s, trusted 0.74 s.

### Why the log covers 40 of the 78 qualified train tasks

From the frozen protocol and code (not an anomaly): `budgets.rollouts.tasks = 40`, and
`Pipeline.rollout` takes `qualified("train")[:40]`. The train manifest is ordered by repository (the order of
`train_repositories` in `configs/learning_pipeline_v1/curriculum.json`, then by `order_key` within each
repository), so the first 40 are funcy 16, sqlparse 16 and isodate 8 (of its 14 qualified). The
python-string-similarity and iniconfig tasks got no rollouts. 78 = 80 manifest tasks − 2 calibration drops
(`prompt_too_long`). `sft_reference` used all 78 tasks; `rft_self` drew only on these 40.

## Interpretation (hypotheses, untested)

- Cheap-verifier errors may still occur with other policies (for example trained arms), other
  repositories, or larger edits. The pilot shows only that they were absent for the base policy on these
  40 tasks from 3 repositories. Testing G1/G3 needs a log with non-zero cheap false positives.
- The synthetic-flip settings were not run, because the abort rule comes first by design. Synthetic flips
  alone would be mechanism evidence (E2-style), not evidence about this verifier pair.
- The low informative-group count (7/40) also limits any group-relative training signal at G=4 for this
  policy and task mix.
- This record does not reopen the stopped partial-label efficiency line. It records that its precondition
  (non-zero cheap-verifier error) was absent here.

## Reproduce

```
python3 pilot_analysis.py --graded ../rollouts/graded.jsonl --out-dir out/   # needs the hash-only log
python3 -m unittest test_pilot_analysis -v                                   # 19 tests, synthetic data only
```

`pilot_analysis.py` is committed byte-identical to the version that produced the outputs (sha256
`4c37ed516fa5d7357ec756272a43ba05fb3abe2cff8717c7078f23722beac049`). Its docstrings mention
`novelty_design.md`, an external design note that is not in this repository. The pre-registered
content of that note is the `PREREG` dict and the summary above.

## Files

- `pilot_analysis.py`: analysis (pre-registration v2 embedded)
- `pilot_result.json`, `pilot_verdict.md`: outputs of the real run
- `test_pilot_analysis.py`, `make_synthetic_graded.py`: unit tests and synthetic-log generator (schema `vare.learning_pipeline.rollout/v1`, `synthetic: true` rows)
