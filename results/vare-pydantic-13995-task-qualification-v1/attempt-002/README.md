# Attempt 002: task qualification stopped at untrusted baseline evidence

The prospective Pydantic #13995 qualification did not pass. The pinned offline grader completed the baseline arm and returned a pytest failure summary, but the grade adapter did not recognize it as an exact-selector failure because pytest appended the failure reason after the selector. No reward was assigned, and mutant-v2 and fixed arms were not run.

The raw grader log and command remain private; only their SHA256 digests are recorded. No task prompt was shown to a model, no held-out task was accessed, and no training or evaluation occurred. This task remains ineligible until a fresh, explicitly frozen qualification run is completed under the corrected exact-selector parser.

The observed grader summary shape motivated a narrow parser change on the integration branch. That change accepts an optional pytest reason suffix after the full anchored selector. The original grader artifacts and failed attempt are preserved unchanged.
