# Task: keep TimesFM padding values safe in fp16

Repair `TimesFmModel._forward_transform` in
`src/transformers/models/timesfm/modeling_timesfm.py`. The configured padding
value can exceed the fp16 scalar range. Preserve the padded value and keep
ordinary normalized values unchanged, without triggering strict fp16 scalar
conversion overflow. Run the provided public smoke test after editing.

The public test includes a CPU shim for a strict fp16 conversion failure
reported by the upstream CUDA test. It is a deterministic compatibility check,
not a CUDA execution result.
