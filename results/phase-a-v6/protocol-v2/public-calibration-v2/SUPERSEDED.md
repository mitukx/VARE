# Superseded provenance note

The v2 public calibration outputs and their original checksum manifest are retained unchanged. A post-run lint-only edit to the calibration script changed its committed SHA-256, so the `calibration_script_sha256` field in the v2 raw records does not match the committed script. Do not use v2 as source-to-generator provenance evidence. The v3 calibration directory was rerun using the committed script whose SHA-256 matches every v3 raw record; v3 is the provenance-complete public baseline/fix calibration.
