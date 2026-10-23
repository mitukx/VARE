# Phase A v6 public-test result integrity

The study controller now accepts an explicit public-test result only when its `classification`, `passed`, and `gradeable` fields form a valid tuple. Contradictions are converted to `runner_integrity_error`, which stops grading before reward calculation. Historical records without the explicit classification field retain their legacy return-code classification path.

The existing protocol-v1 study and its raw outcomes remain frozen. Because the study runner source changed after that run, protocol v1's exact implementation-commit gate prevents replay from the new HEAD. No future inference may use v1; a future study must use a prospective protocol that pins the corrected runner and fresh task identities. No inference or training was run for this change.
