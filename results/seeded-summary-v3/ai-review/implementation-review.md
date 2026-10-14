# Independent seeded-summary implementation review

Executed diagnostics use only the hand-made synthetic fixture task expressions and synthetic output strings. CLI probes replace the actual confirmation generator with a raising trap and model-session constructors with raising traps; neither is called. No training, model generation, or actual confirmation-cohort inspection was performed.

## API provenance bypass: found and corrected

Initial `current_api_binding_probe.py` returned `prospective_pin_present` even with missing or wrong analysis hash tags. The public pure summarizer accepted caller-controlled `prospective=True` without verifying row provenance.

Correction independently confirmed by `current_api_binding_corrected.py`:

- missing row tags reject;
- wrong row tags reject;
- non-Boolean prospective input rejects;
- valid tagged rows return `prospective_rows_bound` and `analysis_file_pin_verified=False`.

This accurately separates pure row-bound computation from CLI/driver filesystem pin checks. Corrected module SHA256 at review: `35088476d53e61f3cb938187157edbf89ce19ece5bc0d744b71d58dec36d2f5a`.

## CLI provenance read race: outstanding at this review snapshot

The reviewed CLI reads progress bytes for hashing, then independently reopens the file using read_jsonl. Another writer can change the file between those reads. The emitted metrics then refer to new rows while progress_sha256 identifies old bytes.

`current_cli_read_race.py` deterministically models that interleaving at the read_jsonl injection point with a complete valid synthetic grid. Its JSON records that a summary was emitted, the reported hash is the old blob hash, the current progress hash differs, and primary metrics differ from the old blob's summary.

Parse the captured byte blob once and compute both metrics and digest from it. Original failure files are preserved.

## Successful actual CLI function probes

`current_cli_adversarial.py` invokes analyze() on synthetic runs:

- prospective and explicitly retrospective summaries succeed with appropriate status;
- existing outputs cannot be overwritten;
- raw progress and legacy summary sentinel remain byte-identical;
- missing entire seed, wrong row tag, and duplicate task ID reject before producing a summary;
- actual confirmation generator/model sessions are never called;
- legacy module/protocol bytes remain unchanged.

These CLI probes were run before the status-name correction; successful row-binding and no-overwrite behavior are unaffected by that additive labeling correction.

## Static metric assessment

New aggregation averages per-fitted-run estimates before averaging tasks and labels base decoding blocks separately. Legacy primary bootstrap and decision objects are carried through from the untouched v2 implementation. Complete-grid validation rejects missing seed/task/sample units before aggregation. Across-run SD is explicitly descriptive and includes sampling noise. The task bootstrap scope is conditional on the fixed fitted runs and observed generations. Legacy pooled curve and any-20 McNemar are separate fields.

No further metric weighting or uncertainty issue was found within that stated estimand and scope. Earlier exhaustive enumeration was not repeated for this implementation review.
