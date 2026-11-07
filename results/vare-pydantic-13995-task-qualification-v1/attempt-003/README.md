# Attempt 003: preserved untrusted adapter failure

The fresh baseline arm ran in the pinned offline grader and completed with the expected frozen-selector failure. The local call adapter then raised `KeyError` because it did not supply the selector field required by the trusted grading function. No trusted grade or reward was written, and mutant-v2 and fixed arms were not run.

The raw grader log and full command remain private. This public record contains only their SHA256 digests and sanitized structural results. No model prompt was exposed, no held-out task was accessed, and no training or evaluation occurred. The baseline is rerun under a fresh freeze with the adapter field corrected.
