# Phase A v6 run-2 public-test classification correction v3

Version 3 preserves the seven-invocation event accounting from v2 and corrects its run provenance fields. In v2, `source_run_manifest_sha256` was incorrectly set to the hash of the run's `SHA256SUMS` file. Version 3 binds `run-manifest.json` and `SHA256SUMS` separately, and also binds the original report. The regression verifies each field against its named, unchanged source file, along with each event's raw trajectory/result hashes.

The original run, v1 correction, and v2 correction remain unchanged. All seven serialized calls are still classified as Colima `container_launch_failure`; no test function ran and the learning result remains NO-GO.
