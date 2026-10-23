# Isolated hash-checked structured edit core

**Status:** library-only prospective integration. No coding-agent rollout, model
inference, grader calibration, training, or held-out assessment was performed.

This change extracts the atomic structured-edit primitive and its tests from
the existing stacked structured-agent branch while leaving the frozen
`run_local_agent.py`, `agent_task_contract.py`, and all historical grader
implementations unchanged. It is intentionally **not** wired into a coding
agent yet; calling this module directly is not a new certified v10/v11 agent
protocol.

`apply_structured_edit()` accepts only an exact allowlisted source path and
hashes of the source preimage and complete workspace revision. It supports
one occurrence of `replace_text` or `insert_text`, writes via a temporary
file and `os.replace`, and emits a deterministic unified diff with hashes.
The tests cover stale hashes, no-op replacements, duplicate search strings,
path traversal and symlink rejection.

**Known gaps:** this module is not a fully concurrent or adversarial filesystem
transaction: path checks and replacement occur at different times, and the
caller must maintain workspace isolation, ownership and locking across the
operation. It is **NO-GO for untrusted concurrent writers**. Before enabling
this inside a live Agent, create a separately pinned source protocol and
verify source-lock compatibility, independent reward semantics and atomic
staging behavior. Do not modify frozen historical protocol locks.
