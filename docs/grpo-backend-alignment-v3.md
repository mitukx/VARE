# Native-sampler backend diagnostic v3

The retained v2 CPU study demonstrated exact score agreement in FP32 and BF16
on 32 sampled tokens. It did not measure quantized alignment or Metal. The v1
q4 CPU arm timed out; it remains an incomplete measurement.

This [prospective protocol](../protocols/grpo_pretrained_sampling_alignment_v3.json)
uses new neutral prompts and seeds on the same hash-pinned SmolLM2-135M-Instruct
weights. Three explicitly separate arms: Linux x86_64 q4 CPU, Apple Silicon
BF16 Metal, and Apple Silicon q4 Metal. Each arm is capped at four completion
tokens, four action segments, one short context group, and 480 seconds of
measurement. The strict 1e-6 tolerance and actual captured denominators remain
unchanged. This is a diagnostic, not a new allocation of Train/Dev/Held-Out
tasks or a model learning experiment.

The workflow probes the requested backend before any checkpoint download.
A virtual macOS runner may have no usable Metal device. Such an arm records
BLOCKED_BACKEND; no CPU fallback or numerical claim. An execution failure or
timeout is incomplete. A completed measurement may report PASS or
NO_GO_NUMERICAL_MISMATCH; a successful workflow indicates the diagnostic
ran/recorded its disposition, not that every backend passed.

The runner records raw sampled token sequences, group score comparisons,
precision and quantized-module inventory, parameter byte digests, resource
time, and runtime versions. No optimizer or task grading occurs. A measured
pass is restricted to the tested model, device, precision, and short context.

To reproduce on a clean registered host, install the pinned requirements
from requirements/grpo-trajectory-cpu-test.txt; on macOS omit mlx-cpu.
Run preflight first, then download and run with HF_HUB_OFFLINE=1 using
scripts/diagnose_grpo_backend_alignment.py. Use the exact protocol and a new
output directory; keep blocked and incomplete outcomes. The provided
workflow applies external download and measurement wall caps.

Status: prospective, no v3 measurement claimed at authoring.

## Completed v3 outcomes

All three arms completed at the measured source. CPU q4 passed; both Metal arms failed numerical alignment. See the [original byte-exact archives and bounded outcome table](../results/grpo-pretrained-sampling-alignment-v3/README.md). No optimizer step was attempted. Automatic v3 repetition is disabled. This diagnostic is not a model training or Metal correctness claim.
