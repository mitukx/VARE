# Run 9 protocol audit correction

Run 9's raw baseline/fix and eight-probe outputs remain unchanged and reproducible, but the run is invalid as a protocol-v4 calibration result. Protocol v4 locks the evaluator, calibration runner, red-team script, and tests by SHA-256. Run 9 used changed evaluator/runner bytes and its calibration script checked only the baseline source hash. Therefore its `calibration_passed` field is descriptive of observed outcomes, not valid acceptance under the frozen v4 protocol. Run 8 has the same limitation. Run 6 remains the last calibration that matches protocol v4's locked implementation hashes.

Protocol v4 is preserved. Protocol v5 locks the current implementation bytes and the updated runner now verifies every required implementation hash before grading. Run 10 is the first calibration eligible for protocol-v5 acceptance.
