# Seed-aware pass@k and progress snapshot analysis

## Result and scope

The frozen format-versus-correctness v2 summary pools outputs from five fitted
runs before applying pass@4. That answers a different question from drawing four
outputs from one fitted model and then averaging its performance across runs.
A hand-made complete grid reverses the ordering of two synthetic arms under
these two definitions. The additive v3 analysis reports the latter explicitly.
It preserves the v2 primary pass@1 bootstrap and decision without recomputation.

Independent AI review also exposed a provenance error in the first v3 CLI:
metrics and their progress-file SHA could describe different bytes. CLI and
driver reports now parse and hash one captured progress blob. These are CPU
analysis correctness results. No model was trained or sampled, and no actual
confirmation-cohort tasks were inspected in the fixtures.

## Immutable sources

| Role | Revision or SHA-256 |
| --- | --- |
| v2 baseline, PR #18 | `3eb6de55251406ed6c8f77eeef34817f41fc5155` |
| Initial v3 CLI with read race | `b918e6ea0e69954c3081fa63e770428785cc7840` |
| Corrected implementation | `3a5a808e71d948cbe531a952344f8d5dee91f625` |
| Unchanged v2 learning protocol | `8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4` |
| Unchanged v2 analysis module | `fca8d5174747c1f3c1a8207d179dde83e9ee7b84ccb7767ddd5a33ae2fb96c15` |
| Separate v3 analysis protocol | `4729ad5294f896f4d6e30fef1535fd28006a7b194b7080d6fe659915f060cc1f` |

The learning protocol, scorer, parser, seeds, data, budgets, and promotion rule
are unchanged. Existing `summary.json` fields retain their original meaning.
`summary.seeded-v3.json` is a separate analysis, not a replacement of the frozen
primary experiment. The v2 model comparison remains unrun in this contribution.

## Estimand and counterexample

The existing [HumanEval estimator](https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py)
for a task with `c` successes among `n` outputs is

```text
g(n,c,k) = 1 - choose(n-c,k) / choose(n,k).
```

VARE v2 calls that estimator with `n=20` and successes pooled across five runs.
For the declared v3 question, each run has `n=4`; compute `g(4,c_s,k)` for each
run, then take equal means over tasks and runs. A nonlinear transformation
generally cannot be moved outside the mean.

| Successes in each run's four outputs | Legacy pooled pass@4 | Mean per-run pass@4 |
| --- | ---: | ---: |
| `[4,0,0,0,0]`, synthetic `format_sft` | 0.6243550051599587 | 0.20 |
| `[1,1,1,0,0]`, synthetic `correctness_reward_rl` | 0.5087719298245614 | 0.60 |

The pooled statistic favors the first arm; the per-run statistic favors the
second. Neither arm name denotes an actual trained model in this fixture.
The pooled statistic is retained as `legacy_pooled_curves`, describing draws
from the finite output pool across models. It is not intrinsically invalid for
that different sampling experiment. The estimator itself is existing work;
the contribution is identifying and testing VARE's aggregation boundary.

At k=1, balanced complete grids have identical rational values under both
aggregations. Reviewer enumeration of all `5^5=3125` success-count vectors
confirmed this equality, with floating differences up to `5.55e-17` in the
implementation. Consequently, v3 copies the legacy primary bootstrap,
decision, and final per-run pass@1 objects exactly. It does not retune a decision
threshold or reinterpret a conditional task interval as a training-seed interval.

## Snapshot defect and correction

The initial CLI performed `progress.read_bytes()` for hashing and independently
called `read_jsonl(progress)` for analysis. A deterministic file replacement
between those calls attached the old SHA to new metrics. In the fixture, the
primary contrast changed from approximately -0.05 to +0.15 while the reported
SHA still identified the old rows.

`read_progress_snapshot()` now reads once, parses that blob, and returns its
SHA. Both the standalone CLI and the driver's final reports use its rows and
digest. Malformed JSON, non-object records, and invalid UTF-8 reject without
rewriting the input. An interleaving test covers prospective and retrospective
CLI paths; a synthetic session test covers the actual driver reporting path.
The retained baseline reproducer fails its metrics/digest correspondence
assertion; the corrected implementation passes all correspondence checks.

This guarantees that the hash identifies the bytes analyzed. It does not lock
a writer, make a filesystem read atomic, or promise that the file remains the
latest version after analysis. A concurrently captured partial record rejects.

## Analysis provenance and uncertainty

- A prospective run pins the separate analysis before arm output, and every
  progress row binds its SHA. Old output without a pin requires explicitly
  retrospective analysis. Existing CLI output cannot be overwritten.
- The pure API validates row bindings and reports `prospective_rows_bound` with
  `analysis_file_pin_verified=false`. Only the CLI/driver, after filesystem pin
  inspection, reports that file check as true. A pin is not a signature or proof
  against an adversary who controls all local assets.
- Missing samples/seeds, duplicate rows, unknown task IDs, duplicate task IDs
  across splits, and Boolean integer metadata reject before averaging.
- The legacy paired bootstrap resamples tasks conditional on these fitted runs
  and observed generations. It does not provide coverage over new training
  seeds. The new run-contrast sample SD is descriptive and includes generation
  sampling noise. The base arm's seed variation is decoding-block variation,
  not independent retraining.
- The old McNemar diagnostic concerns any success among 20 pooled outputs. It
  remains available as `legacy_any20_mcnemar`; it is not primary pass@1 inference.

## Reproduction and retained evidence

Use Python 3.12 or later, Git, and a fresh output directory. The core replay uses
the standard library and pytest 8.4.2, with no optional model packages, network
model downloads, checkpoints, API, or accelerator. Git may fetch the two pinned
source commits if they are absent from the local clone.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest==8.4.2
python scripts/reproduce_seeded_summary_v3.py --output /tmp/vare-seeded-summary
python -m pytest tests/test_seeded_summary_v3.py tests/test_format_vs_correctness_v2.py
```

The replay retains actual baseline/fixed summaries, hand-made tasks and raw
outputs, the expected baseline assertion failures, the CLI interleaving checks,
26 passing new tests, and source/artifact hashes. The targeted suite has 31 tests
including five existing v2 checks. The [result bundle](../results/seeded-summary-v3/)
also retains the broader suite and independent AI review. The source hashes in
its replay manifest identify the implementation tested, even when subsequent
commits change only reports and evidence.

At implementation `3a5a808`, the macOS arm64/Python 3.12.12 broader suite passed
381 tests in 62.46 seconds. Selected new analysis/reproduction files passed Ruff
0.15.12, and the new analysis module passed mypy 1.19.1 with silent imports.
These selected static checks do not claim whole-repository lint/type coverage.
Exact commands and installed optional test dependencies are in
[`environment.json`](../results/seeded-summary-v3/environment.json).

For a completed existing run, without training or modifying its raw progress:

```bash
python scripts/summarize_format_vs_correctness_seeded_v3.py \
  --run /path/to/completed-v2-run --retrospective
```

For future prospective runs, use the existing v2 driver at this revision; it
writes both summaries. Keep one owner per output directory. Resuming legacy
progress without the analysis pin is refused; do not insert a pin after observing
arm outputs and call the result prospective.

The reviewer executed independent synthetic probes, found the API labeling and
CLI snapshot defects, and checked their corrections. Review scripts and failed
attempts are retained. They contain original local scratch paths and are review
records; the portable reproduction entry is the command above. AI review and
CI reruns are not independent human reproduction. No claim is made about model
quality, production incidence, GPU behavior, or recursive improvement.

The next useful measurement is the separately frozen model comparison after its
runtime, decoding cost, and objective preflight pass. This repair makes its
secondary curves interpretable; it does not supply the missing model result.
