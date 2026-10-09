# GSM8K sequence-level DPO development v2 — invalidated

## Decision: protocol provenance failure

The v2 development result is **not valid evidence under its frozen protocol**. The protocol specifies hash ranks 1376–1439 for training and 1440–1503 for validation. The committed `run-2` bundle and the additional local `run-1` bundle instead contain the same rows as development v1: ranks 608–671 and 672–735. They therefore reuse an earlier development cohort rather than the new cohort claimed in the report.

This was independently reconstructed against the pinned GSM8K training cache. The saved v2 runner snapshot calls the row builder with the v1 ranges. The prior auditor also silently supplied those same defaults whenever a protocol omitted machine-readable ranges, so it accepted the v2 cohort while claiming to verify the lock. The old `audit.json` pass is not valid evidence of v2 row provenance.

The outputs recorded four-epoch mean validation preference NLL 0.57436 versus 0.69315 at base, mean token KL 0.43843, and exact match 1/64 for the base and all updated seeds. These are descriptive numbers from a reused cohort, not a valid frozen v2 result. The committed [run-2 bundle](../results/cpu-lm-gsm8k-sequence-dpo-development-v2/run-2/) remains as an explicitly invalidated historical record. The extra run-1 bundle is preserved privately.

## Corrective change

The frozen v1/v2 protocol files remain unchanged as historical records. The runner and auditor now resolve the two legacy protocols to their explicitly frozen rank ranges, and reject missing, partial, overlapping, or incorrectly sized ranges for later protocols. A row reconstruction from the pinned cache rejects the committed v2 rows against the v2 lock. On a temporary copy adjusted to the current manifest inventory, the corrected auditor also rejects at the locked-row comparison.

Future sequence-DPO results need machine-readable rank ranges in the locked protocol and independent agreement among protocol, runner snapshot, retained rows, and auditor. The separate confirmation v2 remains a frozen **non-pass** on its own new rows because mean KL exceeded its 0.5 ceiling. Its outcome is unchanged, but the development-to-confirmation selection history must disclose that v2's development cohort was reused.

## Limits

The local run's three-prompt generation parity check is only a narrow implementation check. Neither the invalidated v2 training result nor the parity check demonstrates free-form model improvement, human preference alignment, broad reasoning, transfer, or capability gain.
