# Retired MLX-LM #1971 v10 operator interface

**Status: superseded before execution by v11.** The v10 protocol and lock are
preserved unchanged, and no v10 operation was executed.

This protocol is **distinct from frozen v9**. Do not edit v9 protocols,
locks, raw calibrations or manifests to make a current checkout look clean.
The v10 operator contract lives in `protocol-v10.json` and the separate
SHA-256 lock `protocol-v10.lock.json`. During review, two defects were found:
its pinned source commit used an attestation that inherited executable Git
configuration, and its sparse checkout omitted test files required by nested
v9 attestation. The v10 runner was removed from the current tree. Protocol v11
pins the corrected source and includes the required `tests/` path.

The v10 proposal accepted only `grade` or `export`. It verified the operator
protocol and lock against independent SHA-256 constants, checked v9 protocol
bytes against pinned Git objects, and verified its three-module Python import
chain against the pinned source commit
`45fd0da592955205e6518c12e9e9b315797541e2`. That pinned operator then
attested the historical v9 grader at
`3f12eefd71d1d2213b37e3a174b446cb5155e979`. These details describe the
retired v10 proposal; its runner is no longer present in the current tree.

Do not use the retired v10 interface for new operations. See
[`mlxlm-1971-v11-operator.md`](mlxlm-1971-v11-operator.md) for the prospective
replacement and its source-only nested-checkout regression.

The proposal also documented Git configuration suppression, no-checkout sparse
materialization, strict child evidence, and no-follow patch snapshots. Review
found the attestation subprocess did not receive the sanitized Git environment
and the nested sparse checkout omitted required test files. The replacement
protocol v11 retains the controls and closes those two gaps.

**Non-claims and limitations:**
- No model, Docker grader, new calibration, SFT, GRPO or held-out comparison
  ran to generate this code.
- The v10 implementation was prospective and operator-facing, *not* a
  replacement of old `vare.mlxlm_1971_repo_task.export_candidate` and
  `grade_workspace` public methods. Those legacy public v9 functions are
  still vulnerable to coherent working-tree substitution if invoked directly.
- The bootstrap v10 CLI itself must be delivered from a trusted source:
  security checks within mutable local code cannot defend against an attacker
  who can edit or replace the bootstrap implementation at invocation time.
- Independent review must assess privilege boundaries, digest pins,
  resource controls and result integrity before any new calibration is run.
- Existing v9 CI checks and any successes on historical Development
  fixtures cannot certify fresh held-out tasks.
