# Run-001 independent re-analysis

`reanalyze.py` re-derives every reported run-001 number from the raw graded logs. It is standard
library only and does not import or copy `../analysis/stats.py`, `vare.eval_stats` or
`vare.learning_pipeline.comparison`. Each record's verdict is recomputed from its per-test
statuses (`verifiers.trusted_full_v1.tests` all `PASSED` and F2P/P2P counts complete); the
logged `reward` is only compared against it. Output: `reanalysis.json`.

```
python reanalyze.py --rollouts ../rollouts/graded.jsonl --out reanalysis.json   # rollout log is hash-only
```

## Integrity checks (all hold)

| Check | Result |
|---|---|
| eval `graded.jsonl` (3 arms), `calibration/admitted.json`, rollout log: sha256 equals `../SHA256SUMS.txt` | yes |
| 42 held-out ids per arm, unique, equal to the admitted held-out list; `sample_index` 0 only; split `heldout` | yes |
| train/held-out admitted ids disjoint; repositories disjoint | yes / yes |
| recomputed verdict vs logged `reward` (126 held-out, 160 train records) | 0 mismatches |
| `sha256(output_text) == output_sha256` (held-out records) | 0 mismatches |
| RFT set restated from the protocol rule (trusted pass, at most 2 distinct outputs per task, lowest `sample_index` first) vs `data/rft_self/*.provenance.jsonl` | identical: 16 examples, 12 tasks, no held-out task |

## Reported vs recomputed

| Quantity | Reported (`RESULTS.md`, `comparison.json`, `analysis/stats.json`) | Recomputed |
|---|---|---|
| held-out tasks | 42 | 42 |
| pass no_update / sft_reference / rft_self | 15 / 14 / 20 | 15 / 14 / 20 |
| Wilson 95% no_update | 0.230-0.508 | 0.22989-0.50834 |
| Wilson 95% sft_reference | 0.210-0.484 | 0.21012-0.48447 |
| Wilson 95% rft_self | 0.334-0.623 | 0.33360-0.62278 |
| rft vs base (both / rft only / base only / neither) | 15 / 5 / 0 / 22 | 15 / 5 / 0 / 22 |
| rft lift; exact McNemar two-sided; Holm | +0.119; 0.0625; 0.125 | +0.11905; 0.0625; 0.125 |
| rft Newcombe 95% | [+0.011, +0.221] | [+0.01120, +0.22089] with the continuity-corrected phi; [+0.0217, +0.2113] with uncorrected phi |
| rft paired bootstrap 95% | [+0.024, +0.214] | [+0.0238, +0.2143] (own RNG, 10,000 resamples) |
| sft vs base table; p; Holm | 10 / 4 / 5 / 23; 1.0; 1.0 | 10 / 4 / 5 / 23; 1.0; 1.0 |
| sft Newcombe 95%; bootstrap | [-0.167, +0.120]; [-0.167, +0.119] | [-0.16652, +0.12038]; [-0.1667, +0.1190] |
| exploratory: rft one-sided p; mid-p | 0.031; 0.031 | 0.03125; 0.03125 |
| exploratory: rft vs sft discordant; p | 8 : 2; 0.109 | 8 : 2; 0.109375 |
| per repository no_update -> rft (tinydb, colorlog, h11) | 6->9, 4->6, 5->5 | 6->9, 4->6, 5->5 (14 tasks each) |
| per repository no_update -> sft | 6->3, 4->6, 5->5 | 6->3, 4->6, 5->5 |
| cheap pass; cheap false positives | 15/15/20; 0/1/0 | 15/15/20; 0/1/0 |
| malformed/unapplied; finish=length | 5/10/5; 4/5/4 | 5/10/5; 4/5/4 |
| train rollouts: samples, passes, tasks with pass, all-pass, mixed, cheap FP/FN | 160, 32, 12, 5, 7, 0/- | 160, 32, 12, 5, 7, 0/0 |

No numeric discrepancy was found. Under the frozen rule (Holm-adjusted exact McNemar < 0.05 and
lift >= 0.10), rft_self vs no_update stays `no_detectable_change` (Holm 0.125).

## Notes (not numeric discrepancies)

* The `Newcombe` figures match only the variant with the continuity-corrected phi
  (`max(ad - bc - n/2, 0)` when `ad > bc`). `RESULTS.md` says "method 10" without naming the variant.
* Scope: this checks the arithmetic from the logs and the logs' internal consistency. It does not
  check that the logged test statuses are trustworthy. The v1 grader parses pytest stdout and is not
  isolated from candidate code (see `../../grader-isolation-audit/`).
* The analysis output kept with the raw run directory contains an extra `train_rollouts` block
  that the committed `../analysis/stats.json` lacks. The block's values equal the recomputed train
  numbers above.
* The committed run files `train_report.json`, `run_meta.json` and `events.jsonl` contain absolute
  local paths from the run host. They are left unchanged here because they are hash-listed evidence.
