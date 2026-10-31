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

## 6. v2: verdict outside the candidate's control, fail-closed `RepoEnv.grade`, A3 calibration

Section 5 records that a candidate running inside the pytest process can forge the v1 verdict.
v2 separates test *execution* (in the sandbox) from *judging* (in the trusted parent).

Files (all under `scripts/`; `src/vare` is not edited, see "Why not in `src`" below):

| File | Role |
|---|---|
| `learning_pipeline_trusted_runner.py` | in-sandbox runner, passed to `python -I -c` |
| `learning_pipeline_trusted_verdict.py` | `grade_isolated_v2`, frame parser and judge, v2 static patterns |
| `learning_pipeline_trusted_repo_env.py` | `TrustedRepoEnv`, a `RepoEnv` subclass whose `grade` defaults to `isolation="required"` |
| `learning_pipeline_grader_calibration.py` | A3.1-A3.10 calibration harness (CLI and library) |
| `configs/learning_pipeline_grader_v2/Dockerfile` | dependencies-only image for the calibration on Docker/Colima |

### 6.1 Verdict channel `hmac_frames_v1`

1. The parent draws a fresh 32-byte key and a nonce per call. The key goes to the sandbox on
   stdin only (never argv, environment or files). The runner reads it with `python -I` before any
   workspace import, then closes stdin and moves its private copy of stdout away from fd 1
   (fd 1 becomes stderr), so candidate `print`/`sys.stdout` cannot write on the verdict channel.
2. The runner collects whole test files (a missing test ID cannot abort the session), keeps only
   the requested IDs plus the two per-call canaries, and executes each test function itself
   (`tryfirst` `pytest_runtest_protocol`). Pytest report hooks, `pytest_pyfunc_call` and
   `conftest.py` hook implementations therefore do not decide outcomes. References to pytest's
   internals are captured before candidate import.
3. It emits HMAC-SHA256-authenticated JSON frames, `start`, `collected`, one `result` per item,
   `finish{exit, n_results}`, and terminates with `os._exit(exit)`. Code that runs after the
   last test (atexit, finalizers) cannot change the exit code.
4. The parent accepts a pass only if every frame authenticates under the call's key and nonce,
   sequence numbers are contiguous, results follow the collected order without duplicates,
   `finish.exit` equals both the exit code implied by the results and the process return code,
   canary `test_a` PASSED and `test_b` FAILED, and every requested ID PASSED. Plain pytest
   output carries no authority. Anything else fails closed with a `verdict_integrity_*` reason
   (`collection_error` when neither canary was collected).

The v1 `BLOCKING_PATTERNS` list is unchanged (it is part of the run-001 scan replay). v2 adds
`BLOCKING_PATTERNS_V2` (introspection such as `gc`/`inspect`/frames, raw fd access, frame-prefix
literals, comparison overrides).

### 6.2 Fail-closed `RepoEnv.grade`

`TrustedRepoEnv.grade(..., isolation="required")` is the default. With no attested backend, a
failed attestation, or an attestation for another backend or configuration digest, it returns
`passed: False`, `reason: isolation_unverified`, `reward: 0`, `executed: False`, and touches
neither the clone nor any interpreter. `isolation="legacy_native"` must be passed explicitly. It
calls the frozen v1 `RepoEnv.grade` and labels the record `backend: native_venv`,
`attested: false`. Every record carries `isolation` with the mode, backend, configuration digest,
full attestation, backend selection attempts, verdict channel and runner sha256.
`select_backend([DockerBackend, BwrapBackend], ...)` returns the first backend whose attestation
passes, and records each attempt.

### 6.3 Why not in `src`

The v1 protocol freeze hashes `src/vare/learning_pipeline/repo_env.py`, `pipeline.py` and
`scoring.py`, and the frozen v3/v4 inventories enumerate `src/vare/**/*.py`. Editing or adding
there would break run-001 verification or silently change an inventory. v2 duck-types the
interface used by `scoring.score_candidate` and `scoring.calibrate`. A future protocol version
can move these modules into `src` together with a new inventory (v5) and protocol hash.

### 6.4 A3 calibration (synthetic fixture)

`learning_pipeline_grader_calibration.py` builds a git repository whose HEAD is the reference fix
and whose tree carries the bug, and then checks:

| Item | Check |
|---|---|
| A3.1 | reference fix passes in the sandbox |
| A3.2 | unrepaired bug fails |
| A3.3 | every mutant of the reference fails |
| A3.4 | empty file and whitespace-only edits fail |
| A3.5 | unknown test ID, empty ID list, and syntax error fail |
| A3.6 | import-time sleep and busy loop end in `test_timeout`, with no process left |
| A3.7 | a positive oracle control passes; candidates that read protected paths (host canary, evidence, other task, reference, `.git`) fail |
| A3.8 | 12 verdict-tampering candidates fail with the static scan on **and** off: the three reproduced audit exploits (argv echo + exit 0, canary-aware exit 1, git restore), forged frames, protocol-hook hijack, report and pyfunc hooks, pytest internals patch, conftest drop, atexit, `sys.exit` at import |
| A3.9 | no file persists outside the workspace, no detached process survives |
| A3.10 | evidence tree hash unchanged; evidence directory not visible in the sandbox |

Residual cases, reported separately and not counted as defended: patching the runner's own
closures through `gc`, and a return value whose `__eq__` is always true (specification gaming,
not channel forgery). Both pass with the static scan off and are quarantined with it on. v2
narrows the channel. It does not make in-process forgery impossible: code in the same
interpreter can, in principle, read the key from memory. The static scan remains a required
layer, and its patterns can be evaded by obfuscation.

Linux (bubblewrap, this host): all ten items `verified`. This does **not** show that the Docker
backend works on the Mac.

Mac (Docker on Colima). Not run yet. `DockerBackend` accepts only `name@sha256:<digest>`
references, and a locally built image has no repository digest until it is pushed, so the image is
pushed to a loopback-only local registry first. From the repository root, at the commit under test:

```sh
export DOCKER_CONTEXT=colima
docker run -d --rm --name vare-registry -p 127.0.0.1:5000:5000 registry:2
docker build -t localhost:5000/vare-lp-grader:v2 configs/learning_pipeline_grader_v2
docker push localhost:5000/vare-lp-grader:v2
IMAGE="$(docker image inspect --format '{{index .RepoDigests 0}}' localhost:5000/vare-lp-grader:v2)"
echo "$IMAGE"   # localhost:5000/vare-lp-grader@sha256:<64 hex>
WORK="$HOME/.cache/vare-grader-calibration-$(date +%Y%m%dT%H%M%S)"   # under $HOME: shared with the Colima VM
PYTHONPATH=src python3.12 scripts/learning_pipeline_grader_calibration.py \
  --backend docker --image "$IMAGE" --work-root "$WORK" --timeout 120 \
  --out results/learning-pipeline-grader-calibration-v2/calibration-colima.json
VARE_GRADER_IMAGE="$IMAGE" PYTHONPATH=src python3.12 -m pytest -q tests/test_learning_pipeline_trusted_grading.py
docker stop vare-registry
```

Exit status 0 and `all_verified: True` are required. The report records the backend, image
digest, attestation and per-item evidence. Before real tasks are used, each repository needs its
own dependencies-only image, and an attestation and A3 run for that image.
