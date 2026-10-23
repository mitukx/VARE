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
path traversal, symlinked roots, symlinked components, and multiply linked
(hardlink-aliased) source files, which could otherwise disclose content from
outside the intended candidate workspace. The root check includes every
ancestor component, so an alias above the workspace is rejected before reads.
Generated diffs preserve CRLF bytes and include Git's missing-newline markers;
regressions replay both per-edit and snapshot diffs with `git apply` and compare
the exact resulting bytes, including bare CR and Unicode line separators.
Allowlisted paths containing ASCII control characters are rejected before
reading or editing; those characters would make raw diff headers ambiguous.
Paths must also equal their canonical POSIX representation; dot components,
repeated separators and trailing separators are rejected before mutation.
Text and anchor uniqueness includes overlapping occurrences (`aa` in `aaa`),
which are rejected rather than silently choosing the first offset.
Git-metadata components and their protected aliases are rejected even if listed
explicitly: case-insensitive `.git`, trailing dots/spaces, NTFS `git~1` and
alternate data streams, and aliases containing Unicode format characters.
Git refuses patches targeting that namespace on protected filesystems.

**Known gaps:** this module is not a fully concurrent or adversarial filesystem
transaction: path checks and replacement occur at different times, and the
caller must maintain workspace isolation, ownership and locking across the
operation. It is **NO-GO for untrusted concurrent writers**. Before enabling
this inside a live Agent, create a separately pinned source protocol and
verify source-lock compatibility, independent reward semantics and atomic
staging behavior. Do not modify frozen historical protocol locks.
