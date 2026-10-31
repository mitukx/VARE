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


## 2026-10-31 — Complete metadata-only Development/Train family overlap review

- **Hypothesis:** The three source-qualified candidates can be compared with prior Development/Train families using frozen identity, family, and changed-path metadata without opening Held-Out material.
- **Evidence:** Review record `results/mlsys-fresh-development-task-qualification-v1/task-family-overlap-review-v1.json` has SHA-256 `bc1da2611436e354f51013a15ca0b5e856726cfb666ee63d7f33416ab3c24af5`; all parent/addendum input hashes passed. The historical catalog has six Development/Train task families. No direct identity or exact family match was found. Transformers #49418 differs from prior Transformers #45852; Ray #66694 has no same-repository prior. MLX #4778 has possible broad gradient/autodiff relatedness to Train #4662. Historical patch changed paths are incomplete.
- **Alternative explanations:** MLX gradient tasks may be related at a broad level, or distinct at the operation-specific family level. Public pretraining exposure remains unknown.
- **Decision:** Do not claim the full cohort is disjoint. Prepare a prospective Development protocol for Transformers #49418 and Ray #66694 only; keep MLX #4778 unassigned pending replacement or explicit risk handling. No split assignment or inference is authorized by this review.
- **Falsification criteria:** New verified family/identity evidence, any hidden Held-Out access, or input hash mismatch.
- **Follow-up:** Verify local model/runtime availability, then freeze allocation, prompts, budgets, and independent grading before a bounded target-model attempt.

## 2026-10-31 — Allocate two fresh Development agent attempts

- **Hypothesis:** The local Qwen3-4B-4bit policy, using VARE's current bounded native-tool controller, can repair at least one source-qualified coding task family it has not previously attempted.
- **Evidence:** The frozen family-overlap review found no prior family overlap for Transformers #49418 and Ray #66694. Both tasks have baseline failures, known-fix passes, and pinned independent test images. Run 19 previously produced a 2/2 Train-task grade, but it is not fresh Development evidence. Public pretraining exposure is unknown.
- **Alternative explanations:** Any success could depend on the controller, task specificity, or pretraining exposure. The small two-task diagnostic cannot establish generalization or a learning effect.
- **Decision:** Assign only Transformers #49418 and Ray #66694 to a prospective Development diagnostic. Freeze one greedy attempt per family, 2,048 generated tokens, no public tests, one independent grader call per candidate, no retries, and no grader feedback. Keep MLX #4778 unassigned.
- **Falsification criteria:** Both attempts produce no independently accepted patch, or either apparent success fails its frozen independent correctness suite.
- **Follow-up experiment:** Inspect all tool traces and grades. If both fail, identify the largest observed failure stage and change one justified factor before spending another fresh family. Do not start RL without multiple verified trajectories and a nondegenerate reward signal.

## 2026-10-31 — Treat the rejected runner launch as pre-inference infrastructure failure

- **Hypothesis:** The local agent can proceed under the frozen capability protocol after supplying the real .git directory required by its pinned-workspace guard, without changing the source, task, model, controller, or policy.
- **Evidence:** The Transformers launch loaded model weights, then raised PinnedWorkspaceError at workspace revision validation. It stopped before chat-template rendering, prompt tokenization, generate_turn, model output, or grading. The source remained at its pinned base hash and diff was empty. Both replacement workspaces now pass pinned Git revision/status and empty-diff checks.
- **Alternative explanations:** The failure could have consumed some model computation during weight loading, but it produced no policy action or task-token forward pass. Its peak memory was not captured.
- **Decision:** Preserve the failed launch and classify it as infrastructure preflight, with zero generated tokens and the original single inference slot unspent. Addendum 1 binds clean real-.git workspaces. No retry is permitted after any generation begins.
- **Falsification criteria:** Either replacement workspace fails the pinned source, HEAD, clean-status, or empty-diff checks; or any prompt tokenization/generation occurs before the corresponding frozen run is recorded.
- **Follow-up experiment:** Repeat resource checks, then execute exactly one generation per task and grade once without feedback.

## 2026-10-31 — Complete the one-shot fresh Development coding-agent diagnostic

- **Hypothesis:** The frozen local Qwen3-4B policy, using the bounded VARE tool controller, would produce at least one independently accepted repair on two source-qualified Development families with no prior task-family overlap found.
- **Evidence:** The one authorized offline greedy attempt on Transformers #49418 generated 126 tokens over four tool turns, exhausted the 4,096-token context, and left an empty diff; its frozen pinned grade failed 4/6 boundary cases while passing 2/6 controls. The one authorized attempt on Ray #66694 generated 399 tokens over five tool turns, exhausted the context, and left an empty diff; its pinned grade failed both target cases and passed 2/2 throughput controls. Both grade calls used their frozen image and selectors once, with zero skips and no model feedback. Raw attempt and grader evidence, resource telemetry, exposure append, and checksums are retained under `results/mlsys-fresh-development-agent-diagnostic-v1/`. The earlier Transformers real-.git guard failure remains separate: it happened before prompt tokenization and consumed no inference slot.
- **Alternative explanations:** The outcome may reflect the 4,096-token context cap, weak source localization, prompt/controller limitations, model capability, task difficulty, or interaction among them. Two greedy attempts cannot distinguish these causes or estimate performance across coding tasks. Public pretraining exposure remains unknown.
- **Decision:** The frozen capability-feasibility rule is **NO-GO** (0/2 independently verified repairs). RL is also **NO-GO**: there is no verified successful trajectory or nondegenerate reward signal. No SFT/RL update or Held-Out access occurred. The result falsifies the frozen one-shot configuration on these two tasks only; it does not establish general model incapability, learning failure under all configurations, or pretraining independence.
- **Falsification criteria:** This decision would be invalidated by a source/hash/protocol mismatch, an unrecorded attempt or grader call, a grader result shown to the model, a skipped frozen case, or evidence that either candidate contained a model-generated nonempty source patch that passed all frozen cases.
- **Follow-up:** Implement and unit-test one justified controller change to address context/tool-use exhaustion. Then source-qualify a new Development family, complete an overlap review, and freeze a new one-shot protocol before any model generation. Never rerun Transformers #49418 or Ray #66694; keep MLX #4778 unassigned until its possible Train-family relation is resolved or explicitly risk-handled.

## 2026-10-31 — Compact only stale, complete tool rounds at the hard context limit

- **Hypothesis:** The two frozen Qwen3-4B Development attempts stopped because complete tool history consumed the 4,096-token prompt budget; dropping oldest complete tool rounds while retaining the original system/task prompt and newest complete tool round will let the controller prepare another prompt without losing the latest observation.
- **Evidence:** Offline re-rendering of both saved transcripts with their exact model tokenizer reproduced every recorded per-turn prompt length. The Transformers history was 4,296 tokens and compacted to 2,765; the Ray history was 4,140 and compacted to 2,795. Both retain the newest complete tool round and fit below the frozen 4,096-token limit. No model generation or grader call occurred. Focused controller tests cover preservation of initial messages/latest results and rejection of incomplete rounds. Replay artifacts are under `results/agent-context-compaction-replay-v1/`.
- **Alternative explanations:** The original failures may also reflect model capability, task difficulty, navigation, or output formatting. A compacted prompt fitting the token cap does not imply that the model will use the freed space correctly. xAI's public Grok Build already documents [manual and automatic session compaction (default threshold 85%)](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/04-slash-commands.md), [durable compaction checkpoints](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-pager/docs/user-guide/17-sessions.md), and a [two-pass summary compaction option](https://github.com/xai-org/grok-build/blob/main/crates/codegen/xai-grok-agent/src/compaction.rs); compaction itself is established engineering practice, not an algorithmic novelty claim.
- **Decision:** Add a deterministic, no-extra-generation fallback that deletes only oldest assistant tool-call rounds with unique IDs and exactly matching tool responses. Keep the system/task messages, newest complete tool round, and full original transcript. If the latest round alone cannot fit, preserve it and stop normally. Do not change any frozen model protocol or claim a capability improvement.
- **Falsification criteria:** Prompt reconstruction mismatch, any loss of the initial task or latest complete tool result, removal of an incomplete/malformed round, or replay that still exceeds the configured cap after removable older rounds exist.
- **Follow-up:** On a newly source-qualified Development family only, freeze and execute the next single-attempt protocol with this controller. Compare the recorded stop stage, prompt lengths, edit validity, and independent grade; do not rerun either consumed family. Keep real-policy learning NO-GO until independently verified success and a nondegenerate reward signal exist.

## 2026-10-31 — Review the next Transformers candidate using metadata only

- **Hypothesis:** Transformers #49458 can proceed to source qualification without repeating a previously exposed Development/Train task family.
- **Evidence:** The frozen protocol `protocols/mlsys_fresh_development_task_family_overlap_v2.json` (SHA-256 `39314381d33e9399e08a7a69053d8f77e49ba5ce2f95934e167e906efee221c8`) bound nine metadata/exposure inputs and the canonical candidate row before review. The result `results/mlsys-fresh-development-task-family-overlap-v2/review.json` verified those inputs and compared eight Development/Train rows. No direct identity or same-family match was found. The only same-repository prior families were 4D attention-mask collation (#45852) and quantized-cache residual flushing (#49418), distinct from the title-derived lazy AutoVideoProcessor lookup family.
- **Alternative explanations:** Some historical changed-file lists are unavailable, and public pretraining exposure is unknown. Metadata cannot establish full source-level disjointness or model novelty.
- **Decision:** Classify the candidate `NO_PRIOR_FAMILY_OVERLAP_FOUND` at moderate confidence and proceed to source qualification only. Leave it unqualified and unassigned; no test, source, Held-out data, or model inference was used.
- **Falsification criteria:** A source-level review showing the same underlying defect/contract as a prior Development/Train family, an input or candidate-row hash mismatch, or evidence of omitted Development/Train exposure would invalidate this decision.
- **Follow-up:** Freeze a source-qualification protocol, then inspect the pinned issue/source/test inputs and establish baseline failure, known-fix success, wrong-mutant rejection, isolation, and valid receipts before any model attempt.


## 2026-10-31 — Freeze CPU source qualification for Transformers #49458

- **Hypothesis:** The pinned public Transformers #49458 regression is reproducible on the fixed Linux CPU image: the merge-base source should fail the exact lazy-resolution assertion, the single-loop wrong mutant should be rejected, and the upstream-fixed source should pass both configuration cases.
- **Evidence:** Metadata-only overlap review v2 found no prior Development/Train family overlap at moderate confidence. The issue body, merge-base archive and source, merged source, upstream regression test, Dockerfile provenance, and offline image identity were hash-checked. The PR merge commit has the pinned merge-base as its first parent and exactly two changed files. No candidate test has run.
- **Alternative explanations:** A failure could arise from environment/import errors rather than the target regression; the wrong mutant could accidentally preserve the tested behavior; a passing public regression could still be too narrow or pretraining-exposed.
- **Decision:** Freeze protocol `protocols/mlsys_transformers_49458_source_qualification_v1.json` at SHA-256 `b4ba593a665828bdbc02b3b383f82423e5e27b3c7abb625dea1e44b0f7e6e524`. Run exactly one arm at a time in baseline → wrong-mutant → upstream-fixed order under the same offline image and limits. Stop if baseline does not fail at the intended assertion. Do not assign the candidate to a split or run a model.
- **Falsification criteria:** Any frozen input or image identity mismatch; baseline import/environment failure or no intended assertion; wrong mutant passes; fixed arm fails or skips either case; resource/isolation limits are violated.
- **Follow-up:** Execute and record the three frozen arms. Qualification can establish only a CPU source-level regression/evaluator contract; a separate overlap review and one-attempt model protocol are still required before any Development inference. Public pretraining exposure remains unknown.


## 2026-10-31 — Correct isolated test-helper import path before source qualification continues

- **Hypothesis:** The first baseline launch was an environment-only collection failure caused by relocating the trusted test overlay outside the source tree; adding the pinned repository `utils` directory to the container's `PYTHONPATH` will let pytest collect the frozen test without changing the semantic oracle.
- **Evidence:** Attempt 001 exited 4 with `ModuleNotFoundError: No module named test_module` while importing the trusted file. Pytest selected zero tests and never ran the target method/assertion. The exact log SHA-256 is `b237bf93972b09aefbaf32e8fa21c5c3051f93ee00a74d2c6878ccda3723008e`. The pinned baseline tree contains `utils/test_module/custom_configuration.py` and `custom_video_processing.py`; their hashes are pinned in addendum 1.
- **Alternative explanations:** The test/source mount boundary could be weakened by overlaying the trusted file into the source tree, or the attempt could be misreported as a target baseline failure. The first would weaken the intended separation; the latter is unsupported because no test case was selected.
- **Decision:** Preserve attempt 001 as `INCONCLUSIVE_TEST_COLLECTION_ENVIRONMENT_ERROR`. Freeze addendum `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/protocol-addendum-1.json` at SHA-256 `b9763deb7f5cd9229da77a001afa38a7a7062d34475a2cf9afa7f0ec3dcdfff5`, which changes only `PYTHONPATH` from `/source/src:/source` to `/source/src:/source:/source/utils` and authorizes exactly one corrected baseline invocation. Do not retry further or run mutant/fixed until the corrected baseline reaches and fails the intended assertion.
- **Falsification criteria:** The corrected invocation still fails to collect; any frozen source/test/image/resource boundary changes; or the test method runs but does not reach the expected baseline assertion.
- **Follow-up:** Push the addendum, run corrected baseline once, preserve its log and receipt, and stop if inconclusive. Keep the candidate unassigned and do not run inference.


## 2026-10-31 — Confirm the intended baseline regression for Transformers #49458

- **Hypothesis:** The unmodified merge-base source eagerly resolves unrelated video processors during AutoVideoProcessor loading, causing the pinned assertion to fail.
- **Evidence:** Corrected baseline attempt 002 used the frozen image and trusted test with only the addendum's `/source/utils` PYTHONPATH correction. It selected one test, failed one assertion, passed zero, skipped zero, and exited 1. `load_attribute.assert_not_called` observed 47 mapping loads. The first video configuration case failed, so the legacy image configuration case was not reached. The exact log is preserved at `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/attempt-002-baseline.log` with SHA-256 `2a06f0a82d534b970fa5900424b008d28e742aef37f486657dfb906eddcc5a96`; receipt SHA-256 `7f029dfc000548f3e88cdd631194527f1d4dd983eaaeb974fa6cd32cba05cfaf`.
- **Alternative explanations:** An import/setup issue could masquerade as baseline failure, or eager resolution could be caused by the added path. The run completed test collection, instantiated the expected processor, then failed specifically because the mock recorded 47 mapping loads; the environment correction only exposed the pinned helper package.
- **Decision:** Accept the intended baseline negative control and authorize the one wrong-mutant run. The candidate remains unqualified until mutant rejection and fixed-source pass.
- **Falsification criteria:** Receipt/input hashes disagree, the assertion failure is not attributable to the 47 mapping loads, or source/test/image/resource boundaries differ from the frozen protocol plus addendum.
- **Follow-up:** Run the single wrong-mutant arm; verify that the target assertion rejects the incomplete fix specifically on the legacy image-config subcase. Do not run upstream-fixed unless this gate is met.


## 2026-10-31 — Preserve the wrong-mutant rejection but stop v1 on missing subcase attribution

- **Hypothesis:** The one-loop wrong mutant is rejected by the pinned test because `_load_backend_class` still traverses the lazy class mapping.
- **Evidence:** Wrong-mutant attempt 001 selected one aggregate pytest case and failed `load_attribute.assert_not_called` after 47 mapping loads (exit 1, zero passes/skips). The output omitted the `unittest.subTest` filename. Static source inspection shows `_load_backend_class` is called when `video_processor_class` is set and its unmodified `VIDEO_PROCESSOR_MAPPING.values()` loop can trigger the assertion on either configuration path. The receipt SHA-256 is `e6809ac56412ac25e5d74a8947a4539fdec34287be709a5ef1861c27e6c2d6ae`; raw log SHA-256 is `efc0512856814536762e88275a10a599204805443b221a9ee7e6129adf3459ae`.
- **Alternative explanations:** Treat the assertion failure as proof that the legacy-image subcase ran; or claim that the test rejected the partial fix without identifying which subcase caused the failure. Neither is supported by the aggregate log.
- **Decision:** Record the mutant as rejected by the target assertion, but classify the v1-required legacy-case attribution as **UNVERIFIED**. Do not run v1 upstream-fixed because its specified wrong-mutant gate was not demonstrated. Freeze a new case-reporting protocol before any additional candidate test.
- **Falsification criteria:** A retained receipt/log that identifies the legacy case, or a case-separated replay showing the required path, would resolve the attribution gap; any source/test/image/limit mismatch invalidates the run.
- **Follow-up:** Use the exact trusted test method with a runner that emits `subTest` parameter names, then execute a fresh baseline → wrong-mutant → fixed sequence under a newly frozen protocol. Do not run model inference or allocate the candidate.


## 2026-10-31 — Freeze case-aware source qualification v2 for Transformers #49458

- **Hypothesis:** Running the unchanged upstream test method under a stdlib unittest result reporter will expose both config subTest outcomes, allowing baseline, incomplete-mutant, and upstream-fixed behavior to be checked separately.
- **Evidence:** V1 logs showed the same target assertion failure but did not expose the `filename` parameter. Source inspection showed `_load_backend_class` runs for both configuration paths and the one-loop mutant leaves its lazy `VIDEO_PROCESSOR_MAPPING.values()` traversal intact. The new runner was checked on a generic two-subTest fixture (one pass, one fail) and emitted both filenames/outcomes; it did not load Transformers source or execute the candidate test. The runner is pinned by SHA-256 `2ec495123c435fd08a8717a993170edce909f81403eb8dd45074154830c6eff7`.
- **Alternative explanations:** Edit the upstream test into a new oracle, infer subTest identity from aggregate pytest output, or continue with v1 fixed. Editing the oracle is unnecessary, the output does not identify a case, and v1's frozen mutant gate was not met explicitly.
- **Decision:** Freeze `protocols/mlsys_transformers_49458_source_qualification_v2.json` at SHA-256 `f62f163be8984323c0d237b82fbaef3cf075a80e5bb49d23fd0fd8328141ee9b`. Reuse the exact upstream test bytes and source trees, changing only the runner to emit per-subTest records. Run fresh baseline → wrong-mutant → upstream-fixed arms, one invocation each; require both named cases to meet each arm's gate. Keep the candidate unassigned and prohibit model inference.
- **Falsification criteria:** Any frozen input mismatch; missing subtest record; baseline or wrong-mutant not rejected by the exact assertion in both cases; fixed arm has any failure/skip; or resource/isolation mismatch.
- **Follow-up:** Commit and push v2 before its first candidate test, then execute the three bounded arms and preserve full receipts. Public pretraining exposure remains unknown; source qualification alone does not authorize a model run.


## 2026-10-31 — V2 baseline fails the pinned lazy-resolution assertion in both config cases

- **Hypothesis:** The unmodified merge-base implementation eagerly loads unrelated processor mappings for both explicit video and legacy image configuration paths.
- **Evidence:** V2 baseline attempt 001 used the case-aware runner and reported `video_preprocessor_config.json` and `preprocessor_config.json` separately. Both reached `load_attribute.assert_not_called` and failed with `AssertionError`: 47 `_load_attr_from_module` calls for the video case and 141 for the legacy-image case. The method ran once, with two subTest failures, zero errors, and zero skips. Receipt SHA-256 `ef5c864eb1bc119a11600bd2fce2b24cc7481bb30f68072cf1073f29b6dc11cc`; log SHA-256 `ecee4766d4c918e36abc71175e1be09748eb263978c1b1ed53dcde5f8bc43566`.
- **Alternative explanations:** A collection/import error could look like an expected negative control, or only one config path could fail while aggregate output hides the other. The runner emitted both named cases and both exact assertion failures.
- **Decision:** Accept the v2 baseline negative-control gate and authorize the one v2 wrong-mutant run. The candidate remains unqualified and unassigned.
- **Falsification criteria:** Any hash/input mismatch; absent config case; error/skip; assertion differs from `load_attribute.assert_not_called`; or nonzero unexpected source changes.
- **Follow-up:** Run the wrong-mutant once and require both named config subTests to fail at the intended assertion without errors/skips before running upstream-fixed.


## 2026-10-31 — V2 wrong-mutant is rejected in both config cases

- **Hypothesis:** The frozen one-loop mutant remains incorrect because `_load_backend_class` still iterates the lazy `VIDEO_PROCESSOR_MAPPING.values()` and resolves unrelated classes.
- **Evidence:** V2 wrong-mutant attempt 001 used the case-aware runner. Both `video_preprocessor_config.json` and `preprocessor_config.json` failed `load_attribute.assert_not_called` after 47 mapping loads each. One test method ran with two subTest failures, zero errors, and zero skips. Receipt SHA-256 `76e78ba63938bfc21013868504960df91a9a01fc7fc2d2093b8078ead3be5d4a`; log SHA-256 `5db740dacb7d9406da01e860fae420cf977125504473bfecaeb7f3d472a53f0c`.
- **Alternative explanations:** A source mutation mismatch or unrelated failure could falsely reject the mutant. The mutant source hash and full-tree digest match the v2 input manifest, and both named cases fail at the frozen assertion.
- **Decision:** Accept the wrong-mutant negative control and authorize the single upstream-fixed arm. The candidate remains unqualified and unassigned until fixed-source passes both cases.
- **Falsification criteria:** Input/source hash mismatch; missing named case; any error/skip; or either case not failing specifically at the target assertion.
- **Follow-up:** Run upstream-fixed once under identical image, test, resources, and isolation; require both named subTests to pass with no errors or skips.


## 2026-10-31 — Qualify Transformers #49458 as a source-level CPU task

- **Hypothesis:** The pinned baseline reproduces the eager-resolution regression, the one-loop mutant remains incorrect, and the exact upstream fix resolves both explicit-video and legacy-image configuration cases.
- **Evidence:** Frozen v2 protocol `protocols/mlsys_transformers_49458_source_qualification_v2.json` (SHA-256 `f62f163be8984323c0d237b82fbaef3cf075a80e5bb49d23fd0fd8328141ee9b`) ran once per arm under the same offline Linux ARM64 CPU image. Baseline: both named subTests failed `load_attribute.assert_not_called` (47 and 141 mapping calls). Wrong mutant: both failed the same target assertion (47 each). Upstream-fixed: both named subTests passed, zero errors/skips/timeouts. Input manifest, receipts, raw logs, report, and SHA256SUMS are retained under `results/mlsys-fresh-development-task-qualification-v3/transformers-49458`.
- **Alternative explanations:** Public issue/test availability and unknown pretraining exposure can explain future model familiarity; the one-method CPU qualification does not represent the full repository or model capability.
- **Decision:** Mark the candidate **QUALIFIED_SOURCE_LEVEL** only. Keep it **UNASSIGNED** pending a separate prospective model-attempt protocol. No model, training, or Held-Out evaluation is authorized by this qualification result.
- **Falsification criteria:** Any input/image/protocol hash mismatch; baseline or wrong-mutant fails outside the intended assertion; fixed case fails; missing case; skip/error/timeout; or evidence of prior Development/Train family overlap would invalidate or narrow the qualification.
- **Follow-up:** Freeze one bounded Development model attempt with exposure logging and an independent grader before inference. Pretraining exposure remains unknown; do not claim capability or RL improvement from this source-level result.


## 2026-10-31 — Freeze one-shot Development attempt for Transformers #49458

- **Hypothesis:** On a distinct, one-file lazy registry-resolution task with two independently qualified configuration cases, the pinned local Qwen3-4B agent can produce an allowed source edit that passes both cases in one bounded attempt.
- **Evidence available before assignment:** The source-level qualification v2 passed its baseline, wrong-mutant, and upstream-fixed gates; its fixed arm passed both named subTests, while the incomplete mutant failed both. The moderate-confidence overlap review found no prior Development/Train family overlap. Earlier Qwen3-4B attempts on quantized-cache and Ray filtering tasks produced no edits and exhausted 4,096 tokens; they remain consumed. The issue and upstream PR are public, so pretraining exposure remains unknown.
- **Alternative explanations:** The model may recognize the public upstream change from pretraining, or may still exhaust context or fail to edit. A one-family success would not establish a success rate or policy improvement.
- **Decision:** Assign this distinct family prospectively to Development and authorize exactly one local greedy attempt with 2,048 total generated tokens, 4,096 context tokens, 12 turns, 300 seconds, and one isolated independent grade. Freeze `protocols/mlsys_transformers_49458_agent_attempt_v1.json` (SHA-256 `4d6561c9d3abe5c85174deca56e77e34f8338d3da3832d4418d82f74e45007ee`) before inference. No retries, grader feedback, public-test access, training, or Held-out use. Current coding-capability and RL decisions remain NO-GO until actual evidence changes them.
- **Pre-inference check:** A no-test Docker mount preflight matched source/test/runner hashes under the read-only layout; zero grader calls have been used. See `results/mlsys-fresh-development-agent-diagnostic-v2/transformers-49458/grader-mount-preflight-001.json`.
- **Falsification criteria:** Any input hash mismatch, failed resource gate, model/runtime failure, no allowed diff, or either named case not passing at independent grade. Every generation attempt consumes the task; do not rerun.
- **Follow-up:** Confirm the frozen protocol is pushed, perform the immediate host preflight, and run attempt 001 once. Public issue and upstream PR: https://github.com/huggingface/transformers/issues/49451 and https://github.com/huggingface/transformers/pull/49458.


## 2026-10-31 — Diagnose the consumed Transformers #49458 attempt and instrument incomplete tool calls

- **Hypothesis:** The one-shot attempt may fail before editing because the model cannot localize the lazy registry path or cannot fit a complete edit-tool payload within the turn budget; the saved transcript, not the final grader status alone, should separate these mechanisms.
- **Evidence:** Frozen attempt `freshdev-agent-transformers-49458-attempt-001` emitted 849 tokens over six turns. It listed the one allowed file, searched `AutoVideoProcessor`, and read lines 360–400 in the legacy `image_processor_type` fallback. The next three model outputs began edit calls for that fallback but omitted the closing `</tool_call>` delimiter. Two re-encoded at 256/256 tokens and one at 255/256; none was dispatched and the candidate diff stayed empty. The single independent grade then observed the unchanged baseline fail both named cases at `load_attribute.assert_not_called` (47 and 141 calls; zero errors/skips). Raw evidence and the transcript-level diagnosis are retained under `results/mlsys-fresh-development-agent-diagnostic-v2/transformers-49458/`.
- **Alternative explanations:** Call all three outputs definitively token-cap truncations despite the 255/256 re-encoded count; call the grader result a candidate rejection; or call the outcome a hardware/transport outage. Those claims exceed the evidence. The common fact is three incomplete, undispatched tool-call outputs; turns 4 and 6 reached the reported cap, and turn 5 ended one token below it after re-encoding.
- **Decision:** Mark the family consumed and classify this as an agent interaction failure with off-target localization, not a verified repair or candidate-patch correctness rejection. Do not retry the task and do not return grader output to the model. Keep coding capability, SFT/RL update, and Held-Out decisions NO-GO. Add telemetry that records incomplete tool-call status and turn numbers, and sharpen the reminder toward a short exact replacement or smallest patch hunk. This change is an engineering diagnosis/reminder improvement, not evidence of increased model capability.
- **Falsification criteria:** A raw trace showing a complete edit call was dispatched, a nonempty candidate patch, or a grade bound to changed candidate bytes would require revising this classification. Any altered protocol/result hash invalidates the corresponding evidence record.
- **Follow-up:** Run the focused controller tests and lint, commit/push the result plus narrow instrumentation to PR #155, then inspect exact-head CI and fix failures. Preserve task consumption; use a distinct qualified Development family and a newly frozen protocol before any future model inference. Public pretraining exposure remains unknown.


## 2026-10-31 — Correct retry wording after exact-head Python 3.13 CI failure

- **Hypothesis:** The initial telemetry/reminder commit should preserve existing user-facing truncation terminology while describing the cause cautiously, because the saved attempt included an incomplete output one re-encoded token below the nominal cap.
- **Evidence:** Exact-head PR #155 run at `09605e254c47bb07a96dec7a8e318a9aae13fd68` completed Python 3.13 with 1 failure, 1,538 passes, and 111 skips. The sole failure was `tests/test_mlsys_4b_pilot_protocol.py::test_truncated_whole_file_call_gets_actionable_retry`, whose compatibility assertion requires the reminder text to contain `cut off`. The failing output and assertions are in Actions job `124413981602`. The initial wording said only “incomplete,” which was semantically cautious but broke that established interface expectation. Locally, after preserving `cut off` as a possibility rather than asserting it as a fact, Ruff and 48 targeted tests passed (the 47 controller-format tests plus the legacy reminder assertion).
- **Alternative explanations:** Weaken or delete the existing test, or claim the CI failure reveals a functional controller defect. The assertion is a valid compatibility contract for the established reminder; the failure is limited to wording and is corrected without changing parser, dispatch, or grading behavior.
- **Decision:** Change the reminder to say the previous call was incomplete and *may have been cut off*, while separately stating the per-response token limit. This preserves the established phrase without overstating the 255/256-token turn. Keep the at-cap versus unterminated telemetry distinction.
- **Falsification criteria:** The targeted legacy assertion or new incomplete-call reminder test failing after this change; or any new exact-head failure not attributable to an unrelated external workflow issue.
- **Follow-up:** Push the wording correction with the recorded CI diagnosis, then wait for every exact-head CI and CPU-evidence job. Inspect all failures and fixes. Do not retry Transformers #49458 or change the NO-GO capability/RL decisions.


## 2026-10-31 — Add an opt-in larger turn cap after incomplete tool calls

- **Hypothesis:** When a model emits an unclosed tool call at the configured per-turn cap, repeating the same cap can make the existing corrective retry unable to finish serialization. Allowing a separately configured larger cap on the immediately following retry may address this output-length failure without increasing the attempt's total-token budget.
- **Evidence:** The consumed Qwen3-4B Transformers #49458 attempt emitted three unclosed edit calls at 256, 255, and 256 tokens under a 256-token per-turn cap. No call was dispatched and the diff was empty. This is evidence of a repeated incomplete-output pattern, not proof that more output budget would have yielded a correct edit.
- **Alternative explanations:** The model may lack task understanding or edit skill; its edit payload may remain malformed at a larger cap; off-target localization may independently prevent a correct repair. Increasing the per-turn limit could also spend the same total budget in fewer turns.
- **Decision:** Add the opt-in `--incomplete-tool-retry-tokens` limit. Omitted configuration preserves prior frozen-attempt behavior. The protocol-driven evaluator forwards the cap only when frozen budgets specify it, and the evidence collector requires the attempt record to match and checks that each expanded turn follows an incomplete call and reminder. An enabled retry remains bounded by remaining total and context budgets; its effective cap is recorded and the reminder reports it. Partial tool calls remain undispatched. No model or grader was run under this change.
- **Falsification criteria:** The deterministic budget-selection or reminder tests fail; a run record omits or misstates the effective cap; total generated tokens exceed the configured total; or a future controlled comparison shows no improvement in complete valid tool calls at matched total-token budget.
- **Follow-up:** Push and integrate this bounded controller change after CI. Do not retry #49458. Only use the option in a separately frozen attempt on a distinct, prospectively authorized task; this decision record does not authorize inference and does not change coding-capability or RL status.


## 2026-10-31 — Bind ungrouped RVL training generations to their VARE task

- **Hypothesis:** Every `rvl_generation` sent to RVL's trainer must match its `Experience.attempt.task`, regardless of whether VARE rollout-group metadata is present. The grouped-advantage path already checks this identity; the ungrouped path may bypass that check.
- **Evidence:** Frozen protocol `protocols/rvl_grpo_training_generation_task_binding_v1.lock.json` pins the baseline source to main `96b3cb86046ccde579caa1c65ccde9f8a2c76e65`. On the deterministic fake-trainer regression, both a changed `prompt_id` and changed prompt text were accepted by the baseline and each reached `train_step`. The candidate adapter now rejects both before generation conversion. The 11 RVL integration-focused test files pass (79 tests); Ruff and `py_compile` pass. Raw logs and source hashes are in `results/rvl-grpo-training-generation-task-binding-v1/`.
- **Alternative explanations:** Normal `RVLGRPOHooks.rollout()` already checks generated task identity, and no evidence shows a historical run contained a mismatched ungrouped record. A malformed direct `Experience` or mutation between rollout and training is therefore a demonstrated boundary case, not established production incidence. The fake trainer does not measure policy optimization.
- **Decision:** Validate RVL prompt ID and exact prompt against the attached VARE task unconditionally in `_to_verified_generation()`. Treat this as E0 adapter provenance integrity; make no claim about task success, policy learning, coding capability, or an upstream RVL defect.
- **Falsification criteria:** Either malformed identity reaches `train_step`, or a valid matched ungrouped/grouped batch is rejected by the new check.
- **Follow-up:** Push a focused PR, inspect exact-head CI and review feedback, and merge only under normal branch protection. Keep the model-learning gate NO-GO until a fresh Development cohort and a separate frozen protocol exist.
## 2026-10-31 — Qualify vLLM #64575 at source level; keep it unassigned

- **Hypothesis:** The frozen five-case request-model regression distinguishes the pinned pre-merge source, a type-policy mutant, and the upstream fixed source under a network-denied CPU harness.
- **Evidence:** After correcting the baseline's Git meaning to the merge commit's direct first parent, baseline attempt 002 produced the two expected type-specific error-message failures and three passing controls. Corrected wrong-mutant v2 produced the single expected message failure and four passing controls. The upstream-fixed arm produced five passing indicators and exit code 0. All source/test/runner/profile hashes matched, and resource limits held. The initial baseline attempt failed pytest logging initialization before collection because Seatbelt denied `/dev/null`; it is retained. Wrong-mutant v1 failed the intended mutation precondition because Pydantic's `Literal['custom']` produced an extra error; it is retained and not accepted as a gate. The excluded server integration test did not run. Raw receipts and logs are under `results/mlsys-fresh-vllm-64575-source-qualification-v1/`; the qualification report SHA-256 is recorded in `research/flagship/state.json`.
- **Alternative explanations:** A successful source-level gate could be misreported as an agent repair or Development assignment; the test overlay could be mistaken for full-repository coverage; public upstream exposure could be mistaken for pretraining independence. None is supported by these results.
- **Decision:** Mark the task **QUALIFIED_SOURCE_LEVEL** only and leave it **UNASSIGNED**. Keep coding capability, SFT/RL update, and Held-Out decisions NO-GO. Public pretraining exposure remains UNKNOWN. Do not claim model capability, an independent repair, or learning.
- **Falsification criteria:** Any manifest/source/test/profile/runner mismatch; baseline or corrected mutant not failing at the frozen error-message assertion; fixed source not passing all five cases; any error, skip, timeout, memory breach, or network/write escape; or a newly identified prior family overlap would invalidate or narrow qualification.
- **Follow-up:** Perform read-only local model/runtime/resource preflight for this distinct source-qualified family. If feasible and still justified, freeze a separate prospective Development assignment, append-only exposure event, one-shot local model attempt, and independent grader before inference. Otherwise retain NO-GO and continue non-inference engineering.


## 2026-10-31 — Freeze one fresh Development attempt with Qwen2.5-Coder-14B

- **Hypothesis:** The pinned local Qwen2.5-Coder-14B-Instruct 4-bit MLX policy, using VARE's bounded tool controller, can produce a minimal source repair for the newly qualified vLLM #64575 request-model task that passes all five independent cases in one greedy attempt.
- **Evidence:** The task is `QUALIFIED_SOURCE_LEVEL` only: the frozen baseline produced the two intended unsupported-type failures, wrong-mutant v2 produced its intended failure, and the upstream-fixed source passed all five cases. Metadata overlap review v4 found no direct prior Development/Train family overlap. The task has never been assigned to a model; public pretraining exposure is unknown. The cached Qwen2.5-Coder-14B snapshot is pinned at `29efdbab55a161237ab1e432a3abaf6c7ae2b477` with 8309494233 weight bytes; MLX-LM 0.32.2 imports its `qwen2` architecture. The pre-protocol resource snapshot passed the 50% memory-pressure, 5 GiB available-memory, 4 GiB maximum swap-used, 15 GiB disk, and estimated working-set gates.
- **Alternative explanations:** A success could reflect public pretraining exposure, task specificity, or the controller rather than learned capability; a failure could reflect source localization, tool serialization, output budgets, or the policy itself. The 14B model's projected working set is close to the current available-memory envelope, so launch-time gates remain mandatory.
- **Decision:** Prospectively assign this one task to Development and freeze one offline greedy Qwen2.5-Coder-14B attempt with a 12 GiB process-tree RSS guard, 600-second limit, 2,048 generated-token cap, and one hidden independent grade. Commit and push the protocol first; recheck resources immediately before model load. Do not retry, train, use Held-Out, or send grader feedback.
- **Falsification criteria:** Any resource/input/hash mismatch, no applied non-empty allowed-file edit, malformed diff, or any of the five independent cases failing; such an outcome remains a task-level attempt result and does not alter the broader NO-GO capability/RL status.
- **Follow-up:** Inspect the one-shot tool trace and independent receipt. If the attempt fails, identify the decisive failure stage and select a justified next model or controller change on a distinct fresh family. If it passes, report one verified repair, preserve UNKNOWN pretraining exposure, and collect more independent families before changing the capability GO/NO-GO decision.


## 2026-10-31 — Preserve Metal access under the offline attempt sandbox

- **Hypothesis:** The initial default-deny Seatbelt profile blocks MLX device discovery, while permitting the required local Mach lookup and IOKit access can restore Apple Metal without opening network access or broadening writable roots.
- **Evidence:** The no-allowance smoke exited before model loading with `No Metal device available`. A tiny `mx.array` addition under the same profile plus `allow mach-lookup` and `allow iokit-open` selected `Device(gpu, 0)` and returned `[2, 4, 6]`. The profile still denies all network access and writes only under the isolated task workspace, home, temp, and attempt-output directories. No model weights were loaded.
- **Alternative explanations:** The model could still fail during weight loading or generation; the smoke only verifies device access. The IOKit allowance is broad at the device-open boundary, so the model remains restricted to the pinned local controller and offline environment.
- **Decision:** Use the hash-bound profile with the two Metal requirements, and include the smoke receipt as a frozen protocol input. Do not run inference until the protocol and assignment are pushed and the automatic launch preflight passes.
- **Falsification criteria:** Any network access, write outside the isolated roots, model weight load during smoke, or failure to reproduce the Metal device/output under the frozen profile.
- **Follow-up:** Execute the single frozen attempt under this profile; preserve any sandbox or runtime failure and do not retry after generation begins.


## 2026-10-31 — Stop the frozen vLLM #64575 model attempt at the resource gate

- **Hypothesis:** The launch-time host check would still fit the pinned 14B model, KV cache, runtime reserve, and required safety headroom.
- **Evidence:** The frozen input and model snapshot hashes all matched. Available memory was 11,993,432,064 bytes; the estimated working set was 11,799,155,161 bytes, and the protocol requires another 1,073,741,824 bytes of headroom. The gate therefore failed by 879,464,921 bytes. The durable receipt is `results/mlsys-fresh-vllm-64575-agent-attempt-v1/attempt-001/launch-preflight.json` (SHA-256 `d33f329af4268f431e7e21bc7aceb81afe7b179a39af79ce44a98e65dc306a2e`). `model_loaded` is false.
- **Decision:** Record **NO-GO / NOT RUN** for this protocol. Do not load weights and do not retry this frozen attempt. The Development assignment remains unexposed to the model; public pretraining exposure remains UNKNOWN. Coding capability and RL update remain NO-GO.
- **Falsification criteria:** Any evidence that model weights were loaded or the frozen attempt command started would invalidate this no-run record.
- **Follow-up:** Complete the PR #160 engineering and exact-head CI work. Any future model attempt must use a distinct fresh task and a separately frozen protocol whose preflight passes.


## 2026-10-31 — Recheck the warmup-cosine CI failure on the next exact PR head

- **Hypothesis:** The Python 3.13 warmup-cosine oracle failure on prior head `b12d4913` may be transient or dependent on its execution environment; the updated test assertion should expose per-command output if the failure recurs.
- **Evidence:** Exact head `a7a2509c4ddc2434053cc23d7a59508cb5b3304d` passed CI run 2035 across all nine jobs, including Python 3.11, 3.12, and 3.13, and CPU-evidence run 743. The same target test also passed locally with Python 3.13.5 when `DOCKER_CONTEXT=colima`. No failure output was generated on the new head.
- **Decision:** Treat the previous failure as not reproduced, not as a diagnosed or fixed defect. Keep the added assertion detail for a future recurrence. Do not claim this test result establishes model capability or learning.
- **Falsification criteria:** Any recurrence on a later exact head requires collecting per-command stdout/stderr and return codes, reproducing under a correctly selected Docker daemon, then fixing the evidenced cause.
- **Follow-up:** Re-run the full workflow on the durable status checkpoint, then await normal PR review. PR #160 has no review submissions or threads and remains draft.
