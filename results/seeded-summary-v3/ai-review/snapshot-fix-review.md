# Independent review: one captured progress blob

Reviewed revision: `3a5a808e71d948cbe531a952344f8d5dee91f625`.

The earlier failing reproducer and JSON (`current_cli_read_race.py` and
`current_cli_read_race.json`) remain separate and unchanged. Corrected checks are
in `snapshot_fix_independent.py` and `snapshot_fix_independent.json`.

## Executed checks

Command:

```sh
/opt/homebrew/Caskroom/miniforge/base/bin/python3.12 /Users/user/Documents/Codex/2026-10-13/var/work/reviewer/seed-stats/snapshot_fix_independent.py
```

All inputs were handmade synthetic tasks and completed synthetic progress. The
actual confirmation generator was replaced by the fixture, and training,
evaluation preparation, and model generation were trapped to fail if called.

- Exercised the actual CLI `analyze` function in retrospective and prospective
  modes. Instrumented `Path.read_bytes` to replace progress with a different,
  valid synthetic grid immediately after capturing its bytes. Both reports
  used exactly one capture; the digest and metrics referred to the original
  captured bytes. The replacement and legacy summary sentinel were preserved.
- Exercised the actual driver `execute` with a completed synthetic grid, the
  canonical analysis pin, and a fake session. Its legacy and seeded summaries
  both matched the captured rows; the seeded primary matched legacy, and the
  seeded digest matched the captured bytes. No training or inference ran.
- Called `read_progress_snapshot` with truncated JSON, a partial final record,
  a non-object record, and invalid UTF-8. All were rejected. Complete JSON
  without a final newline and blank prefix lines were accepted, with exact
  row and byte-digest checks.
- Prospective reports retained `prospective_rows_bound` and verified-file-pin
  status. Retrospective reports retained `retrospective_analysis` and an
  unverified-file-pin status.

## Static inspection and scope

`read_progress_snapshot` parses and hashes the same immutable bytes object.
The CLI obtains rows and digest through this helper once. The driver's final
legacy and seeded reports share the helper's rows, and revalidate the captured
grid before reporting. No new actionable defect was found in this correction.

This verifies correspondence with captured bytes, not the latest version on
disk or an atomic filesystem snapshot. A writer can still mutate progress
after capture; the reports then correctly identify an older captured blob.
Malformed captured records fail instead of being silently dropped. The probe
does not establish filesystem atomicity, prospective timing authenticity,
model behavior, or an experiment result. It does not rerun the full test suite.

## Reviewed source SHA-256

| File | SHA-256 |
| --- | --- |
| `src/vare/format_vs_correctness_seeded_summary_v3.py` | `8b353e2f6acb927d9c973cc624e55a5ce894c5569aec2e03da7927eb10304326` |
| `scripts/summarize_format_vs_correctness_seeded_v3.py` | `a0392c82a35729032086e10c1cf7b20f0a4f597bf8cd16dbf50547735e92b0a9` |
| `scripts/run_format_vs_correctness_comparison_v2.py` | `48f9c429d14950d07b606a768967959ed1dd09e9588ba3285dda0b2bee8e94ac` |
