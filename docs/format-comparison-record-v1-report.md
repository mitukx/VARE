# Retrospective audit of the retained format-versus-correctness record

## Status and provenance

A pre-existing local output directory contains 320 train rows, 17,920 evaluation
rows, an original summary, and 960 checkpoint files for the frozen v2 grid.
This contribution did not execute that training or generation. The original
environment records only `backend=mlx`, `mlx_lm=0.32.0`, and `python=3.12.12`.
The original executor, source revision, hardware, and complete dependency lock
are unverified. Protocol/model strings in progress are consistency checks, not
attestation of the program or weights that produced them.

The [v3 analysis](seeded-summary-v3-report.md) was committed at `5e50d7a` before
this contribution inspected the record's outcomes. The run lacks the v3 pin,
so this is explicitly **retrospective**. Mechanism diagnostics below are post
hoc. Nothing here changes the frozen primary decision or authorizes further
training on these confirmation tasks.

## Retained primary result

The original summary and replay agree exactly: correctness-reward RL minus SFT
is **+0.46875 percentage points**, with a paired task-bootstrap 95% interval
**[-0.078125, +1.015625] points**. The fixed decision is `interval_includes_zero`.
The record does not establish an improvement over SFT or equivalence of the
methods. The interval conditions on the five recorded fitted runs and sampled
outputs; it is not an interval over new training seeds.

| Arm, final checkpoint | Confirm pass@1 | Confirm per-run pass@4 | Pack pass@1 | Pack per-run pass@4 |
| --- | ---: | ---: | ---: | ---: |
| unchanged base, step 0 | 25.234375% | 33.7500% | 42.265625% | 46.2500% |
| SFT, step 16 | 25.937500% | 34.6875% | 41.953125% | 45.0000% |
| format reward, step 16 | 26.250000% | 34.3750% | 42.031250% | 45.3125% |
| correctness reward, step 16 | 26.406250% | 34.3750% | 42.031250% | 45.3125% |
| random reward, step 16 | 26.250000% | 34.0625% | 41.953125% | 45.3125% |

These are regraded archived outputs, not newly generated measurements. Pack and
pass@4 are descriptive secondary results. The unchanged base uses step-0
decoding seeds while updated arms use step-16 seeds; small base/arm differences
are not a paired identical-draw comparison.

## What the signal audit identifies

| RL arm | Groups | Mixed-reward groups | All-zero-advantage groups | Nonzero recorded loss |
| --- | ---: | ---: | ---: | ---: |
| format | 80 | 0 | 80 | 0 |
| correctness | 80 | 8 | 72 | 8 |
| random | 80 | 31 | 49 | 16 |

The format arm's 160 training outputs all satisfy the frozen strict format;
only 24 are correct. Its two rewards are therefore equal in every group, giving
zero standardized advantages. For the frozen loss

```text
L = (A_1 NLL(output_1) + A_2 NLL(output_2)) / 2,
```

that signal contributes zero gradient. The retained format adapters are
byte-identical from steps 1 through 16 within each seed, all LoRA B tensors are
zero, and all 80 saved Adam states have zero first/second moments. The adapter
thus contributes zero LoRA change throughout the saved history. This closes a
specific explanation: the format-only arm supplied no learning signal in this
record. It does not show that format rewards are generally ineffective.

Zero current advantage alone is insufficient to claim a parameter no-op:
Adam can continue moving parameters using existing moments. The correctness
arm's 72 tied groups therefore cannot be called 72 unchanged updates. Independent
checkpoint inspection found continued changes after the first informative
group. Its eight mixed groups establish sparse reward contrast, not the cause
of the overall primary non-pass.

Likewise, mixed random labels alone do not establish nonzero gradient. Of 31
mixed random-reward groups, 15 contain two identical output strings. Their
opposite advantages multiply the same completion NLL and cancel; their recorded
losses are zero. First nonzero LoRA B appears at the first mixed group with
distinct outputs in each random-arm seed. These are mechanism diagnostics, not
a new treatment/control experiment or a claim that sparsity fully explains the
task metrics.

## Reproducible verification

Raw progress SHA-256:
`10860ac27af371a7374bcb7ca38c6bd5171f6d467aef43c3a7c1e95468c73694`.

The [record bundle](../results/format-comparison-record-v1/) includes a lossless
gzip of the 7,382,482 original bytes, original summary/environment, all checkpoint
file hashes, five unique format adapters representing its 80 identical-per-seed
adapter files, and a gzip archive of all 80 format optimizer states. Every file
is below 1 MiB. Other arms' checkpoint tensors remain local; their fuller
diagnostics are independently executed AI review records with explicit scope.

From a checkout containing this contribution, Python 3.12 and pytest 8.4.2:

```bash
python3.12 -m pip install -e . pytest==8.4.2
python3.12 scripts/reproduce_format_comparison_record_v1.py \
  --output /tmp/vare-format-record
python3.12 -m pytest tests/test_format_comparison_record_audit_v1.py
```

No model runtime, checkpoint download, accelerator, or paid service is required.
The replay checks immutable progress bytes, the full metadata grid (including
train/sample seeds), and preserved primary fields. An independent AST/Fraction
oracle regrades all 17,920 outputs, checks strict correctness and format against
the frozen grader, and reconstructs every per-run curve with exact rational
arithmetic. It checks training rewards/advantages and the public format adapter
and optimizer witnesses. It writes new analysis files to a fresh directory.

At audit implementation `148a429`, the full local suite passed **389 tests in
61.20 seconds** on macOS arm64/Python 3.12.12. Eight new audit tests cover the
independent arithmetic/parser, corrupted records, nonzero tensor witnesses,
and optimizer step/momentum checks. Selected new files pass Ruff 0.15.12.
Raw test/replay logs and hashes are retained alongside the record.

Independent AI review regraded the original records, inspected checkpoint
tensors and optimizer moments, checked identical-checkpoint output consistency,
and verified that 963 source files were unchanged. It also found a whitespace
contract error in the first portable audit (`80936f7`): multiple spaces and tabs
must be accepted by the frozen parser. A failing regression was retained and
the correction is `148a429`; no archived outputs were affected by that difference.
AI review is not independent human reproduction. Original-run source/hardware
provenance remains missing even though the archived data can now be reanalyzed
without private paths or a private model checkpoint.

Recorded train-row times sum to 264.6995 seconds and evaluation-row times to
4,819.4981 seconds. They are recorded operation durations, not measured end-to-end
wall time or accelerator hours. They omit some loading work. The matched budget
is 16 optimizer steps per updated arm; RL additionally generates 480 training
completions. Generation tokens and total memory are not recorded. Equal steps
do not imply equal compute or equal amounts of informative gradient.

## Decision

Retain the primary non-pass. Retire this consumed confirmation cohort for future
optimization, parser tuning, and threshold selection. Do not add updates or
select a new intervention using it and then present the same cohort as fresh
confirmation. The next learning study needs a new protocol and untouched tasks,
an independently checked execution revision/environment, and train/dev-only
checks that rewards vary and distinct completions provide non-cancelling signal.
Increasing the format-only budget has no support from this record's saturated
signal. A causal explanation for the correctness-arm result remains open.
