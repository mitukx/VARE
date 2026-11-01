# Run-002 held-out pool: frozen inputs

Byte-identical copies of the inputs recorded before any run-002 environment build or
model run. `SHA256SUMS` pins every file; `tests/test_run002_heldout_allowlist.py` checks
the pins. Do not edit these files; corrections go into a new, separately hashed addendum.

| file | content |
|---|---|
| `allowlist_extension_criteria.md` | frozen repository admission criteria R1-R7 and the sampling seed (sha256 `7ed490f8...`) |
| `content_filter_counts.json` / `.md` | verified per-repo builder content-filter counts, mirror commits, licence survey and API re-check, heavy-repo rule R5 evidence |
| `heldout_pool_passer_ids.json` | instance IDs that pass the builder content filters, per repo, in seeded order (IDs only) |
| `exposure_ledger_ids.txt` | every SWE-smith instance ID already exposed (superset of the run-001 Train and held-out manifests) |

No problem statement, patch, test name or file content of a candidate task is stored here.

Consumers:

* `scripts/run002_heldout_common.py`: candidate repositories, licence allowlist extension, seeded order keys.
* `scripts/run002_build_check.py`: criterion R7 (environment build, size limits, bug-fails / reference-passes).
* `scripts/run002_freeze_heldout.py`: seeded R x k selection and disjointness checks; writes the frozen manifest.
