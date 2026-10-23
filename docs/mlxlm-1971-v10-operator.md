# Prospective MLX-LM #1971 v10 operator interface

**Status: implemented but not used for new evaluation; review and CI required.**

This protocol is **distinct from frozen v9**. Do not edit v9 protocols,
locks, raw calibrations or manifests to make a current checkout look clean.
The v10 operator contract lives in `protocol-v10.json` and the separate
SHA-256 lock `protocol-v10.lock.json`. Its protocol bytes are also anchored
by an independent constant in `scripts/run_mlxlm_1971_v10_operator.py`.

The v10 CLI accepts only `grade` or `export`. It verifies both the operator
protocol and its separate lock against independent SHA-256 constants, and
checks the historical v9 protocol's bytes against pinned Git objects with Git
replacement refs disabled. It also verifies the full three-module Python
operator import chain against that Git source, rather than trusting only the
entrypoint file. It creates an ephemeral detached worktree at the known
operator source commit `45fd0da592955205e6518c12e9e9b315797541e2`,
verifies the implementation file against its Git blob, and invokes the
operator in a separate isolated Python process. That pinned operator in
turn attest-checks the historical v9 grader and runs it from its own pinned
source commit `3f12eefd71d1d2213b37e3a174b446cb5155e979`.

For a trusted operator (no agent-side grader visibility), after setup:

```bash
python scripts/run_mlxlm_1971_v10_operator.py \
  --action grade \
  --workspace /absolute/path/to/staged-candidate
```

For `--action export`, supply `--source-repo` and an output `--workspace`,
and optionally `--patch`. Every path must refer to the intended trusted
runtime location. Keep hidden grader results entirely operator-side.

**Non-claims and limitations:**
- No model, Docker grader, new calibration, SFT, GRPO or held-out comparison
  ran to generate this code.
- This is a prospective trusted **operator-facing** v10 interface, *not*
  a replacement of old `vare.mlxlm_1971_repo_task.export_candidate` and
  `grade_workspace` public methods. Those legacy public v9 functions are
  still vulnerable to coherent working-tree substitution if invoked directly.
- The bootstrap v10 CLI itself must be delivered from a trusted source:
  security checks within mutable local code cannot defend against an attacker
  who can edit or replace the bootstrap implementation at invocation time.
- Independent review must assess privilege boundaries, digest pins,
  resource controls and result integrity before any new calibration is run.
- Existing v9 CI checks and any successes on historical Development
  fixtures cannot certify fresh held-out tasks.
