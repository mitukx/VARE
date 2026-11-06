# Pydantic #14213 task qualification

The prospective Train/Development candidate passed a frozen three-arm qualification using the exact first-parent baseline, a targeted wrong-entry-point-group mutant, and the exact merged source. The baseline and mutant failed the pinned upstream selector with trusted reward 0; the merged source passed with reward 1.

This qualifies the task for a later Train/Development allocation only. The task remains unassigned. No model prompt, Development task, or held-out task was exposed. No training or optimizer update occurred. The frozen held-out NO-GO gate was not changed.

The grader ran in the pinned offline Linux/ARM64 image with network disabled and read-only source mounts. Per-arm trusted grade records retain source, test, environment, grader, container, and log/command hashes. Raw grader commands and logs remain private; the candidate prompt is omitted.

The task uses editable source `pydantic/plugin/_loader.py` from upstream merge `5a97398b92fa508a5db40ce70b2307be2a676946`, baseline first parent `5740091dcd08e4a92940ceae841434a83331cb09`, and selector `tests/test_plugin_loader.py::test_loader`.
