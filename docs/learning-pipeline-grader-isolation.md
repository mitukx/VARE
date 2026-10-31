# Learning-pipeline grader isolation: audit and container design

Status: audit of the frozen v1 grader (`src/vare/learning_pipeline/repo_env.py`, hash-frozen by
`protocols/coding_policy_learning_mlx_4b_v1.json`) plus a proposed fail-closed sandboxed grader
(`scripts/learning_pipeline_isolated_grading.py`; kept outside `src/vare` so the frozen package and the
versioned continuation source inventory stay unchanged). The v1 protocol and its modules are unchanged;
the new grader is for later protocols only. Evidence level: E0 (synthetic fixture, Linux host,
bubblewrap backend). The Docker/Colima backend has **not** been executed for this change.

## 1. Which processes execute candidate-controlled code (v1)

| Step | Process | Candidate code runs? |
|---|---|---|
| generation | MLX policy worker | no (text only; output is persisted before grading) |
| `edit_format.apply_blocks` | pipeline process | no (string replacement) |
| `RepoEnv.grade` -> `python -m pytest` | child of the pipeline, operator uid | **yes**: tests import the overwritten target module, so module-level code runs at collection time; function bodies run during tests |
| `git diff/checkout/clean` | pipeline | no |

Each candidate is executed twice (`cheap_f2p_v1`, `trusted_full_v1`), in the per-repository host
clone under the cache directory, with the per-repository venv's interpreter.

## 2. Access map of the v1 grader child (code reading + reproduced probes)

Reproduced with `scripts/probe_learning_pipeline_grader_isolation.py` on a synthetic repository
(results in `results/learning-pipeline-v1/grader-isolation-audit/probe.json`).

| Resource | v1 behaviour | Probe result |
|---|---|---|
| Environment | `env = dict(os.environ, ...)`, only `PYTHONPATH` removed: all operator variables (tokens, `SSH_AUTH_SOCK`, `HOME`, ...) are inherited | secret variable read: **yes** |
| Filesystem (read) | operator uid, no namespace: home directory, cache (`datasets/` shards with every task's patches and test names, other repositories' clones incl. held-out repositories), run directory | other task's test file read: **yes** |
| Filesystem (write) | same scope | other task's test file modified: **yes**; run-directory evidence file appended: **yes** |
| Reference solution | the clone's `HEAD` is the pre-bug commit and the bug is an uncommitted diff, so `git show HEAD:<target>` is the reference fix | reference read via git: **yes**; a wrong candidate that `exec`s `HEAD:<target>` at import is graded **pass** |
| Persistence | the venv is outside the clone and never reset; `git clean -fd` keeps ignored files | `.pth` hook written to venv `site-packages`; it ran during a later, unrelated grade; still present after `reset()` |
| Processes | `subprocess.run(timeout=...)` kills only the direct child; no session/pid isolation | detached `sleep` survived the grade |
| Network | unrestricted | connection to a host loopback listener: **yes** |
| Verdict | parsed from pytest stdout `PASSED <id>` lines; pytest exit code is recorded but not used | wrong candidate that writes `PASSED <id>` lines for `sys.argv` ids to the saved stdout pipe and calls `os._exit` is graded **pass** (exit 0 and exit 1 variants) |
| Model-visible leakage | prompt holds issue text + buggy target file only (`prompting.py`); generation finishes before grading, so runtime reads cannot reach the model in v1 | not a v1 channel (code reading) |

Consequence for training: any of the three forged-pass mechanisms above would be admitted as a
trusted success, i.e. written into RFT data or rewarded in RL. A text-only policy can emit them.

### Was run-001 affected?

Not by the reproduced mechanisms, as far as the retained logs show:
`results/learning-pipeline-v1/grader-isolation-audit/scan_run001_outputs.py` applies the blocking
patterns of section 4 to the added lines of every applied edit: 0 of 123 train-rollout edits and
0 of 106 held-out edits (37/32/37) match (`static_scan_run001.json`). This bounds only the listed
patterns; it is not a proof that no run-001 verdict was influenced by candidate code.

## 3. Proposed design (container on the Mac; bubblewrap on Linux for tests)

Sequential handoff (already the v1 order, now made explicit):

1. Generate all candidates for a stage; persist `generations.jsonl`; record its sha256 and each
   `output_sha256`; unload the model.
2. Trusted side (no candidate code): parse/apply blocks; stage the task's *buggy tree* (pinned
   commit + bug patch applied) without `.git`; hash the candidate text.
3. Per candidate, a fresh sandbox with an ephemeral copy of that tree, only the target file
   replaced, plus a per-call canary test file. The workspace is deleted afterwards.
4. Trusted side writes evidence. The run directory, cache, datasets, other tasks and the
   reference fix are never mounted.

Sandbox configuration:

| Control | Docker/Colima (`DockerBackend`, flags from `vare.container_isolation.DockerIsolationRunner`) | bubblewrap (`BwrapBackend`) |
|---|---|---|
| image / interpreter | pinned by `@sha256` digest; dependencies only in `/opt/venv`; repo **not** installed | host venv (dependencies only) bind-mounted read-only |
| network | `--network=none` | `--unshare-all` (new net namespace, no interfaces but `lo`) |
| filesystem | `--read-only`, single bind `/workspace` (ephemeral copy), `/tmp` tmpfs | read-only `/usr` + venv, `/work` bind, tmpfs `/tmp`, `--remount-ro /` |
| user / privileges | uid 65534, `--cap-drop=ALL`, `no-new-privileges` | uid 65534 in a user namespace |
| environment | `env -i` with an allowlist (`PATH`, `HOME`, `LANG`, `PYTHONPATH`, ...) | `--clearenv` + same allowlist |
| processes | `--pids-limit`, `--init`, `--rm`, forced removal | pid namespace, `--die-with-parent`, `--new-session` |
| import root | `PYTHONPATH=/workspace` (`/workspace/src` for src layouts), so the workspace copy is what tests import | same |

Fail-closed rules (`grade_isolated` returns `passed: False` with the reason):

| Condition | Reason |
|---|---|
| sandbox tool/image missing, or probe could not run | `backend_unavailable` / `probe_not_run:*` -> `isolation_unverified` |
| any negative probe not denied (env canary, host canary file, evidence dir visible/writable, interpreter tree writable, rootfs writable, default route, non-`lo` interface, host loopback reachable, docker socket, uid 0, >16 visible processes) | `isolation_unverified` |
| attestation from a different backend or sandbox configuration digest | `isolation_unverified_config_changed` |
| added lines match a blocking pattern (process exit, argv, pytest/unittest/conftest/`PASSED`, subprocess/os.system, socket/urllib/http, ctypes, importlib/`__import__`/`sys.modules`, settrace/atexit/`.pth`) | `quarantined_static_flags` (not executed) |
| canary `test_a_<nonce>` not PASSED or `test_b_<nonce>` not FAILED | `verdict_integrity_canary` |
| pytest exit code != 1 (one canary must fail) | `verdict_integrity_exit_code` |
| any requested test not PASSED (incl. MISSING) | `tests_failed` |
| timeout | `test_timeout` |

Admission: before a task is used, `sandbox_calibration` must show reference fix PASS and
unrepaired bug FAIL inside the sandbox (this also checks that the sandbox imports the workspace copy).

## 4. Verified on the Linux host (bubblewrap backend, synthetic repository)

From `probe.json` and `tests/test_learning_pipeline_isolated_grading.py` (29 tests):

* attestation: every probe denied (no env canary, host canary unreadable, evidence dir absent,
  venv and root read-only, no route, only `lo`, host loopback unreachable, uid 65534, 2 processes);
* fail closed: no attestation -> `isolation_unverified`; missing bwrap -> `backend_unavailable`;
  a bwrap configuration without `--remount-ro /` was rejected by the probe (`rootfs_writable`)
  during development, before the flag was added;
* honest fix passes, honest wrong edit fails; all hostile candidates quarantined with the scan on;
* with the scan **off** (containment audit only): argv-echo forgery -> `verdict_integrity_canary`;
  git-restore -> `tests_failed` (no `.git`); reach probe -> no host side effect (no report file,
  other task and evidence untouched, 0 listener connections, no surviving process), 0 workspaces left.

## 5. Not verified / residual risk

* **Canary-aware forgery passes when the static scan is off** (`forge_canary_aware__scan_off`:
  `pass`). Candidate code runs inside the pytest process; in-sandbox checks cannot prove verdict
  integrity. The static scan is a pattern list, and obfuscated code can evade it. Before online RL
  on this grader, a result channel outside the candidate process is needed (for example, each test in
  a fresh interpreter, with results reported through a trusted worker as in
  `vare.isolated_candidate`), or an explicit acceptance of this risk in the protocol.
* The current task's tests are readable by candidate code in the same sandbox (required for
  in-process pytest). The earlier Colima run (`results/isolation-boundary-verification-v1`) records
  the same limitation (`FAIL_SYNTHETIC_CANARY_VISIBLE` for the grader container).
* Docker/Colima backend: argv construction reuses existing tested flags, but this change has not been
  run on the Mac. It needs per-repository images (pinned base digest, dependencies from the lock,
  network only at build time) and a recorded attestation per image before use.
* Probes check the configured sandbox, not the kernel, the VM or the container runtime.
* Run-time and memory overhead of a container per candidate are not measured. The v1 trusted grade
  median is 0.74 s per candidate.

## 6. Run-002 R7 macOS Seatbelt result (2026-11-01)

The Run-002 R7 attempt is recorded separately in
[`results/run002-r7-macos-seatbelt-20261101/`](../results/run002-r7-macos-seatbelt-20261101/).
It uses `scripts/run002_macos_sandbox.py` around the frozen v1 `RepoEnv`; the v1 source hash is
unchanged. The checker completed all 35 candidates on macOS with 12 repositories admitted, 19 with
insufficient valid tasks, and 4 environment-install failures. The attempt covered 903 tasks: 103
passed reference validation, 769 returned `reference_tests_missing`, and 31 returned
`reference_fails`. The held-out freezer returned its refusal code because 12 admitted repositories
are below the fixed 26 x 8 floor. No manifest or downstream learning run resulted.

The R7 install process allows outbound network access only during package installation and writes
within the repository, venv, and task-local sandbox root. Grading denies network access. It keeps the
host `.git` directory outside the test workspace, installs a stub with only the pinned commit/config,
and denies test writes to `.git`. In addition, the host's Git calls set `core.hooksPath=/dev/null`;
the install sandbox also denies writes to `.git`. A macOS integration test used a malicious
`post-checkout` hook and attempted Git-metadata writes; the hook did not fire and the writes were
denied. Exact environment data and the installed package inventories are retained in the R7 report.

This R7 result demonstrates that the configured Seatbelt boundary was exercised for those builds
and grades; it is not a general proof of macOS sandbox strength. Repository tests execute in a
Python process that can read the repository and tests and write elsewhere in the disposable clone.
It does not add an out-of-process verdict channel. The same-interpreter verdict-forgery finding in
the trusted-grader audit remains a blocker for model capability claims and confirmatory grading.
