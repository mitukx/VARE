# Research decision log

This file is append-only. New decisions are appended with dated evidence and falsification criteria; prior entries are not rewritten after outcomes are observed.

## 2026-10-31 — Preserve frozen tool allowlists while the runtime grows

- **Hypothesis:** The v1/v2 Development campaign contracts must remain exactly as frozen; a later runtime tool must not become authorized for those campaigns implicitly.
- **Evidence:** CI failed because its tests required the locked v1/v2 allowlists to equal the runtime's expanded `tools_for("unified_diff")` result. The only additional runtime tool is `replace_text`. Both campaign protocols declare inference unauthorized, and their byte/hash locks are unchanged. The tool itself is constrained by existing exact-context, allowlisted-path tests.
- **Alternative explanations:** The frozen protocol should be edited and re-locked; or the runtime addition should be removed. Editing it would retrospectively change the experiment contract, while removing it would discard an intentional Stage 3 controller capability.
- **Decision:** Preserve both protocol files and hashes. Test that each frozen allowlist equals the current runtime schema with only the later `replace_text` addition excluded. A future campaign that wants this tool needs a separately versioned, frozen protocol before inference.
- **Falsification criteria:** Any other runtime tool addition or removal must make the contract test fail until explicitly versioned; a campaign allowlist containing a tool unavailable to its runner or including an unapproved edit tool invalidates this decision.
- **Follow-up:** After PR #150 CI passes, integrate the correction and version any new campaign schema prospectively before using it.

## 2026-10-31 — Reclassify the deep-qualification Development split after exposure audit

- **Hypothesis:** The three source-qualified Development tasks in deep-qualification v1 can be treated as unexposed because that qualification run itself started no model rollouts.
- **Evidence:** The qualification catalog is pinned at SHA-256 `2b2a1a7c6e64a866dd3732fcd9723999969a1ac69d86f8a4d9f35edf67322c9d` and labels all three tasks unseen. The retained Stage 2/3 exposure ledger at SHA-256 `a2da858454745c86ef7e5162d47552bb659943e03a47442cd03c0e9dff77693a` matches each by exact task ID, source PR, and task family: #45852 has one Stage 2 model attempt, #1912 has two Stage 2 and three Stage 3 attempts, and #1971 has two Stage 2 attempts. The eight attempts yielded no independently verified repair; #1912's applied edit failed 0/14 hidden cases. The frozen source protocol assigns these same three families to Development.
- **Alternative explanations:** Count only successful repairs as exposure; or leave the original “unseen” flags unchanged because the deep-qualification protocol did not itself call a model. Both would ignore that model prompts and task workspaces had already been presented in earlier campaigns.
- **Decision:** Preserve the original source protocol, catalog, and raw evidence byte-for-byte. Append an exposure disposition that marks all three Development tasks previously exposed and makes this split ineligible for fresh Development evaluation or reallocation to Train. No held-out task was used.
- **Falsification criteria:** The pinned task identity, source PR, family, Stage 2/3 attempt totals, or source hashes fail to match the retained records.
- **Follow-up:** Audit the manual candidate queue against the full exposure history, select a replacement Development pool from unexposed families, and independently qualify its baseline, known fix, wrong mutants, and grader boundary before any model run.

## 2026-10-31 — Reject symlinked parents before repository-task grading

- **Hypothesis:** A repository-task workspace can redirect a grader's source snapshot through a symlinked parent even when the final `server.py` path is checked for symlinks; every path component must be opened without following links, and the graded bytes must be the same bytes whose digest is recorded.
- **Evidence:** A pre-fix local reproducer showed that a workspace with `mlx_lm` symlinked to a sibling directory was accepted and its external `server.py` entered the subject snapshot. After the fix, protocol-v11 run 17 rejected the workspace before invoking `grade_calls`. The same run failed the pinned baseline on `dict_arguments`, passed both known-fix controls, matched the fixed source SHA, rejected the fixed-source swap, and verified all nine red-team probes. Raw evidence is in `results/mlsys-repo-eval-v1/run-17/`; run manifest SHA-256 is `921e1260451bf94d9a7d8f19e3f62fed4b5b2c3dfec6d09aeca71ab0756c9ed6`.
- **Alternative explanations:** Rechecking only the final path component could suffice; or the symlink case could be left to the Docker boundary. The direct reproducer showed that workspace export followed the parent before the container boundary, so the grader otherwise received the redirected bytes.
- **Decision:** Read the source through directory descriptors with `O_NOFOLLOW` on each parent and the final file, require a regular file, and use exactly those bytes for source hashing and snapshotting. Accept this as a narrow task-grader integrity fix, not general sandbox certification.
- **Falsification criteria:** A symlink in any workspace path component reaches the grader; graded bytes differ from the hashed bytes; any v1 receipt stops validating; or a required v2 red-team assertion can be forged or omitted.
- **Follow-up:** Push the implementation and run-17 evidence as a focused PR, pass required CI, integrate if repository rules permit, then audit the manual candidate queue for a fresh Development cohort. No model or held-out run is authorized by this calibration.

## 2026-10-31 — Open untrusted workspace targets nonblocking

- **Hypothesis:** A no-follow open still needs `O_NONBLOCK` before `fstat`; otherwise a candidate-controlled named pipe can block host-side source validation before the container timeout begins.
- **Evidence:** Automated PR review on #151 identified the FIFO path and a bounded local subprocess reproduction; repository guidance requires explicit timeout/resource ceilings. The implementation now opens the leaf with `O_NONBLOCK`, rejects non-regular files by descriptor metadata, and adds a subprocess-bounded FIFO regression test. The focused local suite passes 25 tests on Python 3.11, 3.12, and 3.13. Protocol v12 pins this fix and adds a FIFO probe to the red-team suite; run 18 is pending.
- **Alternative explanations:** Rely on the candidate grader timeout or reject special files using a path check before open. The timeout begins only after source staging; a path-only type check introduces a path race and does not bind validation to the opened file.
- **Decision:** Keep no-follow descriptor traversal, add nonblocking leaf open and a regular-file check, and require the FIFO rejection probe to finish before one second without calling the grader. Do not modify the v11 protocol or run-17 evidence.
- **Falsification criteria:** An unwritten FIFO blocks or reaches the grader; v1/v2 red-team receipts stop validating; or v3 evidence can omit or forge the FIFO outcome.
- **Follow-up:** Execute protocol-v12 run 18 on Train task #1904, preserve new raw evidence separately, then push the updated PR head and rerun required CI/review.

## 2026-10-31 — Validate nonblocking FIFO rejection in run 18

- **Hypothesis:** Opening the target with `O_NONBLOCK` permits a regular-file descriptor check to reject an unwritten FIFO promptly without changing the pinned task grader's baseline/known-fix semantics.
- **Evidence:** Protocol-v12 run 18 on clean VARE commit `a7d39bda1da662c18bc7b6738b95f605b679ee30` failed the baseline, passed the known fix with exact pinned source bytes, rejected the source swap, and passed all 10 red-team probes. The FIFO was rejected in `0.00012` seconds with no grader call. The 25-test focused suite passed on Python 3.11/3.12/3.13. Run manifest SHA-256: `24e4ed3e7e5749a3bd19bdf721cc54104fae913d8902352487a9f3f8801b7d4a`.
- **Alternative explanations:** The new test or protocol might be reporting an expected outcome without exercising the real staging path. The red-team probe invokes `grade_workspace` with an actual named pipe and replaces `grade_calls` with a trap; run 18 records rejection and confirms the trap was not entered.
- **Decision:** Accept protocol v12's narrow calibration as **PASSED** and preserve it as separate evidence from the immutable v11/run-17 record. The FIFO fix resolves the automated P1 review finding but does not certify general-purpose sandboxing or model capability.
- **Falsification criteria:** Replaying the pinned implementation fails either task arm; any of the 10 probes fails or is malformed; the FIFO enters `grade_calls`; or run-18 hashes disagree.
- **Follow-up:** Push the FIFO fix and run-18 evidence to PR #151, rerun exact-head checks and review, integrate only under repository rules, then audit the remaining candidate queue for a fresh Development split.

## 2026-10-31 — Classify a mismatched worker nonce as a protocol failure

- **Hypothesis:** A syntactically valid frame with the wrong nonce is a protocol mismatch and must not yield candidate data.
- **Evidence:** `_parse_result` compares the frame nonce before parsing/accepting its payload and returns `(None, "protocol_mismatch")`. The isolated-grader transport taxonomy includes this classification, and the existing transport regression test checks nonce mismatch.
- **Alternative explanations:** Keep the older test label `malformed_result`; this would collapse a well-formed but misbound frame into a syntax error and disagree with the transport taxonomy.
- **Decision:** Update the expected classification and assert the returned value is `None`; keep fail-closed nonce validation unchanged.
- **Falsification criteria:** A mismatched nonce is ever accepted, a payload is returned on mismatch, or the independent calibration no longer distinguishes protocol mismatch from malformed framing.
- **Follow-up:** Run the full Python CI matrix after pushing PR #150 and preserve its receipts.

## 2026-10-31 — One bounded base-model attempt on Train task #1904 passed its independent grader

- **Hypothesis:** On the narrow MLX-LM structured-arguments task, Qwen3-4B with the bounded native-tool controller can produce an independently graded source repair within one greedy attempt.
- **Evidence:** Frozen protocol `protocols/mlsys_task_model_attempt_mlxlm_1904_v1.json` (SHA-256 `7bdef7ee8210dfbd25b3ca811d8bb7c1450b97e4c6015e39d653879aee51ad3d`) authorized one local MLX inference on the already assigned Train task only. Run 19 used 302 generated tokens over five tool turns. The independent v12 grader passed `dict_arguments` and `json_string_control` (2/2) with one grader call and no feedback to the model. The candidate diff is retained with SHA-256 `e95091b6c6ff482234360c01e59ef8a31cd5fd1ff2956ea8d47872decc5d8419`; run manifest SHA-256 `620c985f1ddefe38487a8c00c3eb5095638ffcd5c7667e91b39f97e2f5a242d1`.
- **Alternative explanations:** The result could reflect pretraining exposure to the public PR, task-specific familiarity, or the narrow function-contract grader. There is no fresh Development task, no multi-family cohort, and no policy update. The controller records the MLX package version as `unknown`, although the pre-run environment package metadata was 0.32.4; this is retained as a telemetry limitation.
- **Decision:** Count one narrow independently graded Train-task trajectory as **PASSED**. Append the exposure event without rewriting Stage 2/3 ledgers or the earlier queue audit. Do not claim Development generalization, model-learning, SFT, RL, or algorithmic improvement. The coding-capability milestone and RL update remain NO-GO.
- **Falsification criteria:** Either correctness case fails on the retained receipt; the model, task, source, tool/runtime, or grader hashes disagree with the frozen protocol; generated tokens exceed budget; or the task was not actually assigned Train.
- **Follow-up:** Integrate the queue audit and run-19 evidence through exact-head PR checks. Prospectively qualify a fresh Development cohort without opening Held-out files, then preregister a separate multi-family evaluation before any further base-model screen or policy update.


## 2026-10-31 — Freeze source-level qualification for three fresh public fixes

- **Hypothesis:** Three metadata-sampled, repository-diverse upstream fixes have small CPU regression tests that can be independently replayed as a prospective Development cohort.
- **Evidence:** The deterministic sample contains 71 tasks from six repositories. The selected exact upstream fixes are Transformers #49418, TRL #7668, and MLX #4778. Their base/fixed source files and test-only deltas are SHA-256 bound in `results/mlsys-fresh-development-task-qualification-v1/source-test-input-manifest.json`. Pinned CPU runtimes and test selectors are in qualification addendum 1 (SHA-256 `18010e86e1838a77d8b4f091434c3a9ec1195b6486d8e2dba0d04d7bbea704b6`). The TRL test uses a 20.8 MB public tiny-Qwen fixture pinned to commit `ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda`; only the test’s forward/loss/autograd checks are allowed. No selected regression test, candidate prompt, agent inference, policy update, or Held-out asset has been run or opened.
- **Alternative explanations:** Source-level tests may skip without optional dependencies, require unsupported behavior, fail on CPU, or produce an environmental failure rather than expose the upstream bug. Public pretraining exposure is also unknown.
- **Decision:** Freeze addendum 1 before running any selected test. Require exact baseline failure, exact fixed-source pass of every expected case, zero skips, network-disabled execution, and one task at a time. The baseline source is the predeclared wrong-patch negative control. Do not assign a Development split based on metadata or partial calibration.
- **Falsification criteria:** Any input hash mismatch; baseline not failing the targeted correctness assertion; fixed source failing or skipping a case; resource/network limit violation; or inability to verify the exact fixed source.
- **Follow-up:** Run baseline then fixed selectors sequentially for the three tasks; preserve all results. Only after all three qualify, freeze a separate Development assignment and model evaluation protocol.


## 2026-10-31 — Classify the first Transformers qualification attempt as an environment failure

- **Hypothesis:** Transformers #49418's fixed regression can be evaluated after matching the source commit's minimum Hub and tokenizer versions.
- **Evidence:** Baseline attempt 1 exited 4 during test-module import, collected zero cases, and executed no correctness assertion. The source at the pinned fixed commit requires `huggingface-hub>=1.31.0,<3.0` and `tokenizers>=0.23.1,<0.24.0`; the initial image had `huggingface-hub==0.36.2` and `tokenizers==0.22.2`. The raw log is retained at `results/mlsys-fresh-development-task-qualification-v1/transformers-49418/baseline.log`. A new CPU image pins Hub 1.31.0 and tokenizers 0.23.3; addendum 2 SHA-256 is `9f8f4ce84dbbfcaf9bd9093aa71d00c69034764d5616f1d6946133512746458a`.
- **Alternative explanations:** Treat any nonzero baseline return as the desired bug failure, or proceed to the fixed arm in the same broken environment. Both would confuse missing test prerequisites with the task regression.
- **Decision:** Classify attempt 1 as `INCONCLUSIVE_ENVIRONMENT_FAILURE`, preserve it, and rerun only after addendum 2 is frozen and pushed. Do not count it toward qualification.
- **Falsification criteria:** The corrected no-network image cannot import the pinned source or collect the exact selector; baseline assertions do not fail for the targeted residual-boundary bug; fixed-source cases skip or fail; or any source/test hash differs.
- **Follow-up:** Run exact baseline then fixed selectors under the corrected image; record each case and preserve all output.


## 2026-10-31 — Freeze exact parameterized Transformers test nodes

- **Hypothesis:** The pinned Transformers regression is represented by six `parameterized.expand` node names, and selecting those exact nodes will execute the intended boundary assertions.
- **Evidence:** Attempt 2 under the corrected dependency image imported the test module and reported no match for the unexpanded method; zero test cases were selected and no assertion ran. The six labels and order are frozen in addendum 3 (SHA-256 `ca36eb3796dd49589316f5ac4ff47e19fe21003f5d8e9449f3930499d6deb08e`) from the selected test delta.
- **Alternative explanations:** Count a pytest selector error as a baseline bug failure, or run the entire `CacheTest` class. The log shows the requested node was absent; running a broader class would exceed the frozen scope.
- **Decision:** Preserve attempt 2 as `INCONCLUSIVE_SELECTOR_MISMATCH`. Retry only the six exact parameterized nodes; retain the 600-second aggregate budget and all isolation rules.
- **Falsification criteria:** Any node name/hash mismatch, fewer or more than six collected cases, no target assertion failure on baseline, or any fixed-source skip/failure.
- **Follow-up:** Execute baseline six-node selector, then the matching fixed-source selector if baseline assertions execute.


## 2026-10-31 — Isolate Quanto's lazy extension cache for Transformers qualification

- **Hypothesis:** The six Transformers #49418 parameter cases can reach their frozen residual-boundary assertions when Quanto's lazy CPU extension build has a bounded writable cache, while the task source/test inputs and container root remain read-only.
- **Evidence:** Attempt 3 selected all six exact nodes but failed at `os.makedirs(.../optimum/quanto/library/extensions/cpp/build)` with `EROFS` before the expected equality assertion. A pinned runtime v3 image adds `build-essential` and Ninja. A runtime-only preflight compiled and loaded the exact Quanto extension with `MAX_JOBS=1` under the existing 2.5 GiB memory ceiling, using a 256 MiB tmpfs at only the extension cache path. Docker defaults that tmpfs to `noexec`; an explicit `exec` option was necessary for loading the compiled shared object. Addendum 4 is frozen before the regression retry.
- **Alternative explanations:** The preflight could be mistaken for a successful regression test; compile pressure could invalidate the task resource limit; or a writable whole-root mount could hide unintended filesystem writes. The preflight mounted no task files, ran no test, used one compiler job within the existing memory/CPU limits, and permits writes only to `/tmp` plus the exact 256 MiB package cache.
- **Decision:** Preserve attempt 3 as `INCONCLUSIVE_ENVIRONMENT_FAILURE`; it is not baseline evidence and does not authorize the fixed arm. Retry the baseline only under addendum 4. Keep source and trusted test mounts read-only, networking disabled, root read-only, uid 65534, and all prior test limits.
- **Falsification criteria:** The extension cannot compile/load inside the bounded cache; any task source/test hash differs; the expected baseline assertion does not execute and fail; or any fixed case fails/skips or limits are violated.
- **Follow-up:** Run the frozen six-node baseline once. If and only if the expected assertion fails, run the exact fixed arm; then proceed sequentially to TRL and MLX. No Development assignment or model inference until all qualification gates pass.


## 2026-10-31 — Accept Transformers baseline negative control and correct the fixed-run invocation

- **Hypothesis:** The base source for Transformers #49418 retains the quantized-cache residual-boundary bug, while the exact fixed source passes the same six frozen cases.
- **Evidence:** Baseline attempt 004 under addendum 4 ran all six exact nodes: four failed the target assertion `layer.keys.numel() == expected_residual_length * 2 * 16` with excess residual keys, and two passed. The first fixed invocation was rejected because two command arguments used `CacheTest.test_...` instead of `CacheTest::test_...`; the log contains no case progress or pass/fail summary, so zero fixed cases executed. The source and trusted test hashes remain unchanged. Addendum 5 (SHA-256 `e16c19fb20ea09e4a559b6e3d8353dd6ebef046c3d0cf9ed7b8744847104ec86`) freezes the exact-selector retry.
- **Alternative explanations:** Treat any baseline nonzero exit as evidence, count the malformed fixed invocation as a fixed result, or hand-retype selectors again. Baseline output locates four failures at the target assertion; pytest’s fixed-arm output is an argument-selection error with no test summary.
- **Decision:** Count attempt 004 as the valid expected baseline failure. Keep the malformed fixed invocation inconclusive and unscored; do not rerun the baseline or change the tests. Construct the one fixed-run argv from addendum 5’s node list.
- **Falsification criteria:** Any source/test hash mismatch; the fixed invocation selects other than six nodes; any fixed node fails or skips; or the frozen container/resource limits are violated.
- **Follow-up:** Run the exact fixed-source nodes once. If all six pass, Transformers #49418 qualifies at source level; otherwise preserve the failure and leave the task unqualified. Then run TRL #7668 and MLX #4778 sequentially.


## 2026-10-31 — Preserve a fixed-arm launcher failure before test startup

- **Hypothesis:** The frozen fixed-source test remains runnable when the Docker wrapper passes only the pytest arguments after its configured Python entrypoint.
- **Evidence:** A protocol-driven launcher selected the six frozen IDs correctly but passed the leading `python3` token a second time after `--entrypoint=python3`. The container exited before pytest startup; zero cases were selected. The raw output and attempt record are retained.
- **Alternative explanations:** Count a container start as a test attempt or relax/alter the frozen selectors. The output says Python could not open the duplicate `//python3` script and contains no pytest output.
- **Decision:** Treat this as an inconclusive launcher error; no fixed result is observed and the frozen A5 test authorization remains pending. Keep the exact test node strings unchanged; pass only the pytest argv tail.
- **Falsification criteria:** The corrected container fails before test collection, selects other than six exact nodes, or any fixed node fails/skips.
- **Follow-up:** Run the one fixed-source arm under addendum 5; qualify only if all six cases pass with zero skips and exact inputs/limits match.


## 2026-10-31 — Qualify Transformers #49418 at source level

- **Hypothesis:** The selected fix resolves the frozen quantized-cache residual-boundary bug across the six specified parameter cases.
- **Evidence:** On exact base source, baseline attempt 004 failed the target residual-length assertion in 4/6 cases and passed 2/6. Under the same pinned runtime v3, test-only delta, selectors, and offline limits, fixed attempt 003 passed 6/6 with zero skips in 25.27 seconds of pytest time. Source/test and patch hashes matched the frozen manifest. Paired summary SHA-256: `75bfa210ef7ce8184ade0d5c98c92047d6eab918e6e662d2f12a3ec07d28ea4e`.
- **Alternative explanations:** The fix could pass only because of runtime differences, changed inputs, skips, or broader tests masking the regression. Both arms used the pinned runtime/test file and selector list; source hashes were verified; no skips occurred. This remains a small historical regression suite and public pretraining exposure is unknown.
- **Decision:** Mark Transformers #49418 `QUALIFIED_SOURCE_LEVEL`. Do not assign it to Development or claim model capability. Keep TRL and MLX pending.
- **Falsification criteria:** Recomputed input hashes differ; the retained baseline log does not show the four target assertion failures; any fixed case is failed/skipped; or the exact runtime/selector protocol was not used.
- **Follow-up:** Qualify TRL #7668 and MLX #4778 sequentially before considering a separate Development assignment protocol.


## 2026-10-31 — Preserve the TRL #7668 pair without relaxing its frozen baseline gate

- **Hypothesis:** TRL #7668 restores gradient-checkpointing kwargs for both reentrant and non-reentrant checkpointing, and its exact test can exercise only the pinned tiny fixture.
- **Evidence:** The exact base selector ran two cases; both raised `TypeError` when the tested function rejected `gradient_checkpointing_kwargs` before entering the context body. No test assertion or forward/loss/autograd operation ran on baseline. The exact fixed source passed both cases; those cases performed the test’s two tiny fixture forward/loss/autograd checks, with no generation or optimizer. Source/test/fixture hashes matched the frozen manifests. Paired result: `f2187cb5d31a6aa5b1bfceb174e220eae5589de91e1ad1e1ce0b1ce95579a47c`.
- **Alternative explanations:** Treat any baseline test failure as meeting the negative-control gate, or retroactively broaden the rule. The frozen acceptance rule requires a targeted assertion failure, and baseline raised before assertions.
- **Decision:** Preserve both arms as useful source-level evidence, but mark TRL #7668 `NOT_QUALIFIED_FROZEN_ACCEPTANCE_GATE`. Do not alter the gate after outcome or assign the task to Development.
- **Falsification criteria:** Reproduction shows any baseline assertion executed or a fixed test used a different fixture/source; either would contradict this classification.
- **Follow-up:** Run MLX #4778 under addendum 1. The current qualified count remains 1/3.


## 2026-10-31 — Qualify MLX #4778 at source level

- **Hypothesis:** The selected fix corrects the complex-valued power VJP conjugation for the frozen CPU test case.
- **Evidence:** On the pinned baseline CPU image, the single exact test executed the JVP check and failed the target VJP assertion `mx.allclose(vjp_z, cotangent * mx.conj(dz), atol=1e-4)`. The pinned fixed CPU image passed the same test. Source, test, image, and test-delta hashes match the frozen records. Paired qualification SHA-256: `29eeb28838404a8052ab129b52b547651d65c5bd3e8ea45059e4728718fee7ee`.
- **Alternative explanations:** A CPU/Metal difference or altered test/source could explain the result. Both images are CPU-only, Metal-disabled, and use the same frozen test tree and selector; source/test hashes were verified. The task is one mathematical case and says nothing about model capability.
- **Decision:** Mark MLX #4778 `QUALIFIED_SOURCE_LEVEL`. Keep it unassigned and preserve the historical pretraining-exposure limitation. Together with Transformers this gives 2/3 qualified tasks; TRL remains unqualified under the baseline-assertion gate.
- **Falsification criteria:** Source/test or image identity mismatch; baseline target assertion does not fail; fixed target fails or skips; or execution violates the frozen resource/network policy.
- **Follow-up:** Freeze a deterministic replacement-task selection protocol before inspecting any unselected source/test inputs; qualify that task before considering a Development split.


## 2026-10-31 — Freeze replacement-task selection before source/test review
- **Hypothesis:** A deterministic metadata-only rule can select one repository-diverse replacement for TRL #7668 from the existing frozen sample without outcome-informed source/test screening.
- **Evidence:** The selector is frozen in `protocols/mlsys_fresh_development_task_replacement_selection_v1.json` (SHA-256 `db6e5b7ece581d0f51733e2536f91b1ff0ad4a6737bcb10206cc563e74ccf94b`). Its sole input is the 71-row sample `results/mlsys-fresh-development-metadata-review-sample-v1/review-sample.json` (SHA-256 `810fb823781f3bc44a0272db230cc3d65b43751c415e53229c81104a6183baad`). No unselected source or test contents were used to define eligibility.
- **Alternative explanations:** Metadata may not predict a runnable or well-targeted task. This protocol does not claim task qualification, exposure independence, or model capability.
- **Decision:** Apply the ranked selector exactly once. If the first eligible task cannot satisfy the frozen qualification gate, record that outcome without substituting a lower-ranked task. Keep all existing task outcomes and the exposure ledger unchanged.
- **Falsification criteria:** Any input sample hash mismatch, non-deterministic selection, source/test inspection before selection is recorded, or post-selection change to the eligibility/ranking rule.
- **Follow-up:** Record the selected metadata row; then freeze the task-specific baseline-first qualification addendum before opening the selected source/test files.


## 2026-10-31 — Select Ray PR #66694 by the frozen metadata rule

- **Hypothesis:** Applying the precommitted ranked rule selects a repository-diverse replacement without selecting on source/test outcomes.
- **Evidence:** `results/mlsys-fresh-development-task-qualification-v1/replacement-selection.json` (SHA-256 `716ba5ab3ad6820ab933425d54d6de66a153632e4ba71774e4f55a61038a9e95`) records Ray PR #66694 at rank 1 of 12 eligible rows from the exact 71-row sample. The selection used metadata fields only; source/test files were not opened.
- **Alternative explanations:** The row is only a metadata candidate; its test may be environment-dependent, incomplete, or unsuitable for a CPU-only source-level pair.
- **Decision:** Select Ray PR #66694 as the sole replacement under the frozen protocol. Leave it unassigned and unqualified until the source-level gate is completed. Do not inspect or substitute lower-ranked rows.
- **Falsification criteria:** The sample/protocol hash does not match; rerunning the deterministic selector yields another row; or evidence shows source/test content influenced selection.
- **Follow-up:** Freeze the task-specific qualification addendum, then inspect only the selected PR's hash-pinned changed paths.


## 2026-10-31 — Freeze Ray #66694 source-review and qualification gates

- **Hypothesis:** Ray PR #66694 can be assessed through a source-only review followed by a preregistered baseline, wrong-mutant, and fixed-source test sequence without exposing a task to a model.
- **Evidence:** Addendum 1 (SHA-256 `9d19502e1ef6ac009f95aa986ffdce8cfdb352c8e6216d11f9fdc462567dcda9`) pins the base, PR head, merge commit, allowed path set, source-review boundary, CPU/offline resource ceiling, and assertion-based acceptance gate. It explicitly authorizes no test execution.
- **Alternative explanations:** The metadata-listed tests may not isolate the task or run under a small CPU-only environment.
- **Decision:** Review only the pinned changed paths; freeze exact tests, input hashes, runtime, and one wrong mutant before executing any test. If a gate fails, keep the replacement unqualified and do not select a fallback.
- **Falsification criteria:** Any unpinned source/test is read, test execution precedes its own freeze, an assertion gate is relaxed after outcome, or resource/inference boundaries are crossed.
- **Follow-up:** Conduct the authorized path-limited source review and prepare a second addendum before testing.


## 2026-10-31 — Freeze Ray #66694 regression suite and wrong mutant

- **Hypothesis:** The multi-node span regression is detected by the four task-changed test cases, including a case that rejects a plausible mixed-clock implementation.
- **Evidence:** Addendum 2 (SHA-256 `22f64298f6b05037ca0ac7e61398745294637bb3be4c720402295ff4403d9918`) pins the selected source hashes, the exact four test node IDs, the compatibility-adapted test file (SHA-256 `945e034c6609bc9cf6cd2ae35704f6a3366bc0598547c16a444c542e784dfe3c`), the upstream patch, and a one-expression wrong-clock mutant (SHA-256 `ddbd0679b8b5e6bb0356fd9befd5273ba8d1a9d58b1466eac5e8f90b30b00040`). The exact Ray PR merge-commit wheel is pinned for runtime construction.
- **Alternative explanations:** The adapter or dependency runtime could mask source behavior; therefore the same test file, dependencies, and limits are used across all arms, and the runtime digest must be frozen before any run.
- **Decision:** Build the runtime only. No baseline, mutant, or fixed tests may run until the image identity is recorded in a separate addendum.
- **Falsification criteria:** Any test/source hash mismatch; the mutant is not rejected by the target assertion; the fixed arm does not pass all four cases; a skip occurs; or network/resource limits are violated.
- **Follow-up:** Build the exact-commit wheel runtime and freeze its digest before the first test execution.


## 2026-10-31 — Correct Ray #66694 pretest artifact binding

- **Hypothesis:** The exact source-diff evidence can be stored in a review-clean compressed form without changing any frozen test input or gate.
- **Evidence:** Addendum 2 (SHA-256 `22f64298f6b05037ca0ac7e61398745294637bb3be4c720402295ff4403d9918`) initially referenced the pre-archive manifest SHA. Erratum (SHA-256 `b8a74db2ee832ad8f47a070fca5f041dfe9d9515bf4c5f7a2e6a029b7bcdb684`) binds the current manifest SHA `1631baa877ac8e3f0f260ebf6ebfd9e3ce4ea9328c556cf4be31b5dcdd29279a` and the gzip archive plus its decompressed content SHA `425ba97673e9f90a681a0f30e95b148acd983118893a0012a65d0a326da5b8ae`. No test ran and no selected source/test content changed.
- **Alternative explanations:** A stale pointer could go unnoticed and later weaken evidence reproducibility. All current state references are recomputed from the committed artifacts.
- **Decision:** Preserve Addendum 2 unchanged; use the erratum as the authoritative manifest/archive binding. Its only effect is artifact path/hash bookkeeping.
- **Falsification criteria:** The gzip content hash differs from the original exact four-path diff, or any test/mutant/source hash, selector, or acceptance criterion changed.
- **Follow-up:** Freeze the runtime image identity before test execution.


## 2026-10-31 — Freeze CLI-only retry for Ray runtime build

- **Hypothesis:** The exact pinned Ray runtime can be built unchanged after removing the Docker CLI option unsupported by the installed client.
- **Evidence:** Attempt 001 log SHA-256 `0a0033a83aaf69fddd35047b16a66734164121ebedf2ca91c57fe16fee286518` shows the CLI rejected `--progress=plain` before the builder started; no image or test ran. Addendum 3 (SHA-256 `07cfef610b9140684f2f50b6bc3e34f3a3fb643882ad98a14904e10edcdb3cec`) keeps every build input and only removes that flag.
- **Alternative explanations:** The runtime package install could still fail due to platform/dependency incompatibility.
- **Decision:** Permit one retry with the identical Dockerfile and build context and no `--progress` option. Do not execute any runtime probe or test until the resulting image digest is frozen.
- **Falsification criteria:** Any build input changes, the second build fails, or tests start before image identity is recorded.
- **Follow-up:** Build and capture the image digest and raw build log; freeze them before testing.


## 2026-10-31 — Freeze canonical wheel filename for Ray runtime retry

- **Hypothesis:** Pip will accept the already verified exact Ray wheel when its canonical PEP 427 filename is preserved in the build context.
- **Evidence:** Build attempt 002 (SHA-256 `69ddf6562d322269f08b6b656c3b3e50a4d71b02d131a6e1a76493a357fc0569`) verified wheel bytes `73bfbeed219e33c6521e750c3eeac921fcaaded8eace2698fe29c254e6590946` and failed only at pip filename validation. Addendum 4 (SHA-256 `8a161b0a157b970722ce00fded9031e06ba3d91791e2c876ca057343feb75429`) changes the copied wheel basename and PEP 508 file URL; it pins the corrected Dockerfile hash `9da37a8fb8c60e513e895437bced793250284d650ed5634ed298bdda80bc5c3b`. No image or test was produced.
- **Alternative explanations:** Other dependency or platform constraints may still prevent the runtime from building.
- **Decision:** Permit one build-only retry with this filename correction. Keep the same wheel bytes, base image, package version, source/test inputs, and acceptance rules.
- **Falsification criteria:** Any wheel/hash change, build failure, or test launch before the image digest is frozen.
- **Follow-up:** Build once, preserve the log, and freeze the image digest before test execution.


## 2026-10-31 — Freeze Ray runtime identity and authorize source-level arms

- **Hypothesis:** A single image built from the exact Ray merge commit can replay the frozen multi-node wall-time tests under CPU-only offline limits.
- **Evidence:** The arm64 image ID is `sha256:04c1b652294c61d1536d0a4b89d8d288b3f8499a3cc83f8d1b5175587660598b`. Its exact-commit wheel SHA-256 is `73bfbeed219e33c6521e750c3eeac921fcaaded8eace2698fe29c254e6590946`, Dockerfile SHA-256 `9da37a8fb8c60e513e895437bced793250284d650ed5634ed298bdda80bc5c3b`, and successful build log SHA-256 `924a33ce4e367222598143ab3d71e648fc5f143b904b4d17cfa8367c215892b2`. Addendum 5 (SHA-256 `ca77f6954d42d8d2bc46c55c19b879a2e60133fe3f797b0609c26e5be6857ab3`) binds these with the test/source/mutant hashes and command boundary.
- **Alternative explanations:** Runtime imports or hidden setup requirements may fail; the import-only preflight runs before any task tests and does not count as a task result.
- **Decision:** Permit one import-only preflight, then exactly one baseline attempt; run the wrong mutant and fixed arm only if their frozen gates permit. Do not retry any arm or change test inputs without a new addendum.
- **Falsification criteria:** Image or mount hash mismatch; preflight failure; target assertion absent on baseline/mutant; any fixed failure/skip; or network/resource boundary violation.
- **Follow-up:** Verify image ID and mount hashes, run preflight, and proceed sequentially.


## 2026-10-31 — Ray #66694 baseline negative control passes

- **Hypothesis:** The pre-PR implementation mishandles cross-node timing because the worker-local monotonic clocks have different origins and no shared Unix anchor.
- **Evidence:** Baseline attempt (SHA-256 `4e2097a80e4ce3afec5623fe47270d0d4f24541b02815ea81709c09632d8a58f`) ran all four frozen selectors on image `sha256:04c1b652294c61d1536d0a4b89d8d288b3f8499a3cc83f8d1b5175587660598b`. The two throughput controls passed; the multi-node span assertion observed 10690.0 instead of 12.0, and the builder-anchor assertion observed None. Zero skips; no collection/setup error.
- **Alternative explanations:** A mismatch in source overlay or test harness could produce a false negative. All mount/test hashes matched before execution, and the import preflight passed on the same image.
- **Decision:** Count the baseline as a valid target negative control and permit the frozen wrong-mutant arm. Do not run fixed unless the mutant is also rejected.
- **Falsification criteria:** Baseline hashes or selector set differ, targeted assertions did not execute, or a case was skipped.
- **Follow-up:** Run the predeclared mixed-clock mutant on the same selectors and runtime.


## 2026-10-31 — Wrong-clock mutant rejected by Ray #66694 regression tests

- **Hypothesis:** A plausible implementation that adds the worker-local monotonic end time to the shared Unix start will fail the multi-node correctness case.
- **Evidence:** Mutant attempt (SHA-256 `5ba305e5cc1643cdaf55c9275182734d2bf335d8309eb67a063a779a1df2ace8`) ran all four selectors on the same pinned image and test overlay. The target multi-node span assertion observed 10810.0 instead of 12.0. Dataset throughput also failed (75 instead of 100); the other two cases passed. Zero skips and no environment errors.
- **Alternative explanations:** The mutant could be rejected only by an unrelated case or changed test inputs. The predeclared target assertion itself failed; the additional dataset failure is explicitly recorded as collateral.
- **Decision:** Count the wrong-mutant negative control as rejected under the frozen gate and proceed to the one fixed-source attempt.
- **Falsification criteria:** Source/test/image hashes disagree, target assertion did not execute, any case is skipped, or the failure is a setup/environment error.
- **Follow-up:** Run the exact fixed source once using the same image, test overlay, selectors, and limits.


## 2026-10-31 — Ray #66694 fixed arm passes the frozen gate

- **Hypothesis:** The exact PR source corrects the cross-node wall-time calculation and records a shared Unix anchor without regressing the two frozen throughput controls.
- **Evidence:** Fixed attempt record SHA-256 `673eeaa1d57505b8d1e957177500d174f652ee6547866a7c46737c16cfb418ab` binds four selected cases, zero skips, exact source hashes, test hash, image `sha256:04c1b652294c61d1536d0a4b89d8d288b3f8499a3cc83f8d1b5175587660598b`, and exit code 0. Raw log SHA-256 `20b00d9cc4a3409a3582614799f69f2a82d46a590612cb448479055fd731933d`. Baseline and wrong-mutant gates were previously met; their records and the collateral mutant failure are retained.
- **Alternative explanations:** A source-level CPU regression pass does not establish that a coding model can discover or repair the issue, that the task is independent of pretraining, or that the task family is disjoint from earlier exposures.
- **Decision:** Count Ray #66694 as `QUALIFIED_SOURCE_LEVEL`. The replacement cohort is 3/3 qualified (Transformers #49418, MLX #4778, Ray #66694); preserve TRL #7668 as unqualified under its original frozen gate. Do not yet assign tasks or run inference.
- **Falsification criteria:** Any mismatch in frozen source/test/image identity, a skipped case, a fixed-arm failure, or a baseline/mutant target assertion not executing.
- **Follow-up:** Freeze and execute the task-family overlap review, then freeze Development allocation and inference protocol before a target-model attempt. Keep Held-out data untouched and pretraining exposure labeled unknown.


## 2026-10-31 — Freeze metadata-only task-family overlap review

- **Hypothesis:** The three source-qualified candidate families can be checked against prior Development/Train exposure using identity, issue-family metadata, and changed paths without consuming Held-Out task contents.
- **Evidence:** Protocol `protocols/mlsys_fresh_development_task_family_overlap_v1.json` is frozen at SHA-256 `6273f27dcb8cba457ecd39c50d916a5a841a29c51066c8039ebd1e5f54d2394d` and binds eight inputs before reviewing historical exposure metadata.
- **Alternative explanations:** Public pretraining may include any upstream issue or patch regardless of VARE task exposure; missing historical metadata can make family overlap indeterminate.
- **Decision:** Authorize metadata-only comparison for Transformers #49418, MLX #4778, and Ray #66694. No split assignment, model inference, test execution, or Held-Out access is authorized.
- **Falsification criteria:** An input hash mismatch, use of Held-Out metadata in semantic comparison, or any source/test/prompt read outside the frozen scope invalidates the review.
- **Follow-up:** Execute the frozen comparison, retain candidate-level rationale and limitations, then decide whether the cohort can be allocated prospectively to Development.


## 2026-10-31 — Add candidate identity metadata to frozen overlap inputs

- **Hypothesis:** The original selection protocol and deterministic review sample supply the upstream PR/title mapping needed to compare the qualified candidates with historical exposure metadata.
- **Evidence:** The parent overlap protocol bound source/test summaries and exposure files but did not bind these two selection-metadata inputs. No historical exposure rows had been semantically reviewed.
- **Alternative explanations:** The original selection metadata may still omit the precise family description; if so, mark the comparison indeterminate rather than infer from source/test contents.
- **Decision:** Freeze addendum `protocols/mlsys_fresh_development_task_family_overlap_v1_addendum_1.json` at SHA-256 `5d7a44df1cf27bbef1030c2c51dde1f89fe71361949a71d6404d379c22a8f0f2`. It adds two identity-only inputs; overlap rules, exclusions, and authorization remain unchanged.
- **Falsification criteria:** Any parent or supplemental input hash mismatch or any use of held-out task content.
- **Follow-up:** Verify both protocol layers and all input hashes, then conduct only the permitted Development/Train metadata comparison.


## 2026-10-31 — Bind provenance for the historical exposure comparison

- **Hypothesis:** The prior Development exposure append can be reconciled against its directly cited Stage 2/3 source ledger and frozen task catalog before candidate-level overlap classification.
- **Evidence:** Addendum 2 `protocols/mlsys_fresh_development_task_family_overlap_v1_addendum_2.json` at SHA-256 `283296307bdacf3ebedbcd56274542ba1d5630644c2817b6166cf94d19b1efb5` pins both provenance inputs. The parent protocol and addendum 1 already freeze the candidate metadata and review criteria. No candidate-level classification or split decision has been made.
- **Alternative explanations:** The historical ledger may not contain enough issue-family or path detail to support a disjointness conclusion; if so, report `INDETERMINATE`.
- **Decision:** Continue only after verifying every parent/addendum input hash. The Dev/Train-only filter and Held-Out exclusion remain unchanged.
- **Falsification criteria:** Any hash mismatch or use/output of a Held-Out identifier or content invalidates the review.
- **Follow-up:** Reconcile the frozen candidate identities and paths with verified Development/Train exposure metadata.
