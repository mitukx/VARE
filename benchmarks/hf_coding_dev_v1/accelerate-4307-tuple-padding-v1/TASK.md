# accelerate-4307-tuple-padding-v1

Splitting tuple inputs across process ranks with padding enabled should preserve the tuple container and return the final rank’s padded slice. The current code raises on an uneven split. Correct this behavior while leaving list and tensor inputs unchanged.

Editable file: `src/accelerate/state.py`.

This candidate task is not frozen for model inference. See `task.json` for source, license, execution and limitation provenance.
