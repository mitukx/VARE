# accelerate-4354-multi-cpu-barrier-v1

A CPU process group must synchronize on CPU even when the host also has an accelerator. The current wait path supplies accelerator device identifiers to a CPU barrier and fails on such machines. Correct the CPU barrier call without changing accelerator-backed process groups.

Editable file: `src/accelerate/state.py`.

This candidate task is not frozen for model inference. See `task.json` for source, license, execution and limitation provenance.
