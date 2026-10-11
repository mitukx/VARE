# urllib3 #3352 task and grader feasibility v1

**Decision:** GO for local historical-task and grader calibration (E1); NO-GO for model training, capability claims, or confirmation use. The public issue and its fix make this a development-only task. Both one-shot model attempts produced no applicable patch. This report records a task/evaluator path and a negative feasibility screen, not an improvement result.

## Task selection and boundary

The task is a narrow urllib3 URL-path regression from [issue #3352](https://github.com/urllib3/urllib3/issues/3352), evaluated against the immutable upstream base commit `166f66faa5ef23e255f654e8c6b66ad8590fe630` (tree `0929cc7c1f30a972cd280fcd41faef3c80aa84bd`). The known fix is public at [commit `d150cea`](https://github.com/urllib3/urllib3/commit/d150cea52b37f4633a0dff4e26339c51d23d987b) and [PR #4952](https://github.com/urllib3/urllib3/pull/4952), so the task is retired from any future confirmation pool. The pinned urllib3 source has an MIT license. The task statement paraphrases the issue; it does not redistribute the report author's issue prose under urllib3's license.

The model receives only the frozen task prompt and the relevant pinned source excerpt. It does not receive the task descriptor, evaluator, test vectors, known-fix source, or gold patch. The evaluator and protected assets are outside the candidate workspace. The candidate workspace is created from the source revision with no Git remote and a version shim; the actual Python package code is exercised through `PoolManager.urlopen` and a local HTTP server.

Candidate comparison:

| Candidate | Fit for this run | Decision |
| --- | --- | --- |
| urllib3 #3352 | Small Python source tree, clear request-line contract, pinned MIT source, executable base failure and known-fix success, no container required | Selected for development calibration; public and contaminated, so never confirmation evidence |
| SWE-bench / SWE-rebench V2 | Broader issue diversity and F2P/P2P conventions, but public patches/tests, per-row source licensing, and heavier environment requirements; SWE-rebench V2 contains public gold fields | Useful for later task discovery after task-by-task rights and isolation review; not used here |
| Django historical tickets | Objective regressions exist, but larger ORM/database environments and public fixes make them less suitable for this small Apple Silicon calibration | Not selected |
| BugsInPy / Defects4J | License uncertainty for BugsInPy; Java 11 and project setup overhead for Defects4J | Not selected pending rights/runtime work |

The dataset/licensing and contamination review is an independent AI subagent assessment, not legal advice or human review. References: [SWE-bench dataset guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md), [SWE-bench evaluation checklist](https://github.com/swe-bench/experiments/blob/main/checklist.md), [SWE-rebench V2 card](https://huggingface.co/datasets/nebius/SWE-rebench-V2/blob/main/README.md), [Django #33899](https://code.djangoproject.com/ticket/33899), [BugsInPy license issue](https://github.com/soarsmu/BugsInPy/issues/94), and [Defects4J requirements](https://github.com/rjust/defects4j/blob/master/README.md).

## Locked evaluator contract

The frozen task lock is [`benchmarks/historical/urllib3_3352/protocol.lock.json`](../benchmarks/historical/urllib3_3352/protocol.lock.json), protocol `urllib3-3352-local-http-feasibility-v1`. It declares two fail-to-pass cases (`//v:h`, `//v:h?key=val`), three pass-to-pass paths, the immutable source and fix revisions, model revisions, one attempt per model, decoding settings, and claim limits. The prompt hash is `110304935a1b643c21db2927534401acf2e6bcece1be21c86bcda5924a159270`; the source-context hash is `2512623c6340fdd5a3b98efbe8512d2fe481e0816129e26851d07cb0157d71e3`.

The grader:

- verifies task identity, source/tree/workspace identity, no Git remotes, task/evaluator/script hashes, and allowed candidate path;
- statically restricts candidate edits to a narrow URL-normalization region in `HTTPConnectionPool.urlopen`; invalid edits are rejected before candidate code executes;
- runs the actual urllib3 request path and verifies exact HTTP request lines and HTTP 200 responses;
- puts the read-only candidate checkout in macOS Seatbelt, denies access to the user's home/work/evaluator paths, and permits only the grader's ephemeral localhost endpoint;
- checks exact pinned urllib3 code objects on the socket-connect stack, limits CPU time, wall time, request-line size, and captured output, and checks protected hashes after every case;
- returns failure for missing cases, integrity mismatches, identity mismatches, invalid source scope, or an empty diff.

The static policy intentionally narrows the task and may reject equivalent refactors. It is not a general-purpose coding-agent sandbox. macOS rejected `RLIMIT_AS`; there is no enforced memory cap, so this evaluator does **not** claim complete resource containment. Its integrity lock is a local trust anchor, not a signed external attestation. Pin the VARE commit and asset hashes when publishing or independently replaying the result.

## VARE task identity binding defect

The new task path exposed a VARE catalog contract defect: rollout selection previously trusted `Task.metadata["engineering_task_id"]`, while verification independently trusted `Attempt.metadata["engineering_task_id"]`. A task dispatched as A could therefore use the prompt/workspace/grader for B. A paired regression constructs a second always-pass task and confirms its checks accept the first task's workspace. Against pre-change VARE HEAD `b3204926b3cae291f7c13e3134444c71a3a63172`, both the rollout and verifier identity regressions fail with `DID NOT RAISE`. The fix derives the catalog ID from `Task.id` and rejects conflicting metadata claims both before workspace creation and before grading. The focused after-change suite passes all four tests, including the original successful rollout/verifier control and the deadline test. The baseline log is [`task-identity-baseline-tests.log`](../results/urllib3-3352-evaluation-v1/task-identity-baseline-tests.log).

This enforces equality inside the catalog adapter; it does not cryptographically authenticate the upstream task dispatcher or protect against a caller that constructs an entirely different `Task.id` before dispatch.

An independent code audit exercised that boundary: a caller that supplies a different valid catalog `Task.id` with matching metadata selects that task, and the verifier accepts a caller-supplied `workspace_path` containing a passing workspace for the selected task. These are not cross-task metadata overrides; the adapter has no signed dispatch identity or workspace-origin token. The current campaign runner supplies catalog-created tasks and rollout-created attempts, but this adapter should not be treated as tamper-resistant against arbitrary callers. The audit also prompted an explicit verifier regression for both task-level and attempt-level conflicting metadata claims.

## Calibration and adversarial evidence

The pinned baseline fails exactly the two frozen regressions with `LocationParseError`; all three pass-to-pass controls succeed. The already-published fix sends the exact expected request line in all five cases. The final current-source replay is `results/urllib3-3352-evaluation-v1/calibration-v10.json`; raw grader-output hashes are retained there. The known-fix pass calibrates the grader only; it is not agent discovery evidence.

The adversarial suite attempted protected-file read/write, external network access, timeout, output flooding, and the earlier direct-socket fake-200 reward bypass. All six candidates failed closed, protected hashes remained unchanged, and task-identity and forged-baseline controls were rejected. In the final suite most malicious source edits were rejected by static validation before execution; those particular cases do not by themselves demonstrate Seatbelt denial. The independent reviewer separately reproduced rejection of the previous direct-socket attack. A socketpair slow-drip test confirms the request-line reader respects an absolute deadline; the deadline is not extended by periodic bytes. The latest reviewer and its exact request are retained under `results/urllib3-3352-evaluation-v1/review/`.

The evaluator previously accepted a candidate that opened a socket directly, returned a fake `200`, and exited without calling urllib3. That attack was retained as a failed result; the revised grader rejects the candidate before execution because the source change is outside the allowed method. This history is why the present task is described as narrowly calibrated rather than generally secure.

## Local model screen

Environment: Apple M1 Pro, 32 GiB unified memory, macOS arm64, Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, offline cached snapshots, float16 on Apple MPS. No NVIDIA GPU, distributed run, paid API, or external compute was used. Both models are Apache-2.0 at the exact pinned revisions below. The model protocol allowed one greedy attempt, `max_new_tokens=512`, seed 0; no retry or prompt change was made.

| Model | Pinned revision | Output | Task result | Load / generation | Peak process RSS | MPS driver allocation |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 411 tokens; not an applicable diff; `git apply --check` rejected it; candidate diff empty | The empty workspace fails F2P 0/2 and passes P2P 3/3. This is the unchanged-base score, not a score of applied model code. | 1.838 s / 56.256 s | 3.97 GB | 1.55 GB |
| TinyLlama-1.1B-Chat-v1.0 | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | 512 tokens; generated procedural instructions, not a diff; candidate diff empty | The empty workspace fails F2P 0/2 and passes P2P 3/3. This is the unchanged-base score, not a score of applied model code. | 3.778 s / 61.190 s | 4.84 GB | 11.32 GB |

The combined model input hash was `0e3ef4fb4481f54c851c4014ce38455291253d14773a65289684e086433336ac`. Qwen's saved result is `results/urllib3-3352-evaluation-v1/model-qwen.json`; TinyLlama's is `model-tinyllama.json`. Raw generated text, empty candidate diffs, evaluation outputs, and SHA-256 manifest are retained beside them. Since neither output was a valid patch, “cost per valid attempt” is undefined; the measured generation time is not a cost estimate for successful coding.

This is one greedy response per model on one public development task. It does not estimate success probability, generalization, preference variance, or a model's coding competence. A known-fix success is evaluator calibration. A base-model failure on this one prompt/task pairing does not imply a general inability.

## Research gate and next study

**Training decision: NO-GO.** This run is one public, contaminated task and has no untouched train/dev/confirmation split, task-level base variance estimate, or positive base feasibility. The two outputs are invalid, so there is no evidence that SFT or a VARE intervention can produce an accepted update here. No training experiment was run.

For a future study, compare at least unchanged-base, matched SFT, and one predeclared VARE-selected intervention on the same permitted training tasks and update budget. Freeze a separate confirmation task composition before training; exclude all public known-fix calibration tasks and task families with known model exposure from confirmation. Require an independent grader, a meaningful base-success floor and reward variance on development tasks, valid task/action contracts, verified update/save/reload, model/policy provenance, and multiple seeds. Set a minimum practical task-success gain and compute abort rule before opening confirmation. Estimate power from task-level paired outcomes and the frozen minimum effect; do not use the two single-task outputs here to estimate variance or power.

## Reproduction

From a VARE checkout with the frozen assets, macOS `sandbox-exec`, Git, and the noted Python runtimes:

```bash
PYTHONPATH=src python3.12 scripts/calibrate_urllib3_3352_task.py \
  --output /tmp/urllib3-calibration

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3.12 -m pytest -q \
  tests/test_urllib3_3352_evaluator.py tests/test_environment_catalog.py

PYTHONPATH=src python3.12 scripts/adversarial_test_urllib3_3352.py \
  --base-run /tmp/urllib3-calibration/pinned-base/preparation \
  --output /tmp/urllib3-adversarial
```

For model screens, use a new output directory and only a locally cached exact snapshot. The frozen script forces offline mode and refuses an existing output directory. Do not repeat these consumed model/task attempts or use this task for confirmation.

## Review and limitations

An independent AI subagent reviewed the source and attack surface at VARE HEAD `b3204926b3cae291f7c13e3134444c71a3a63172`. The report found no confirmed current bypass, verified the known-fix path and the slow-drip helper, and identified the local protocol lock as an unsigned trust anchor. This is an AI review, not human review; no outside human review or independent human reproduction has occurred.

Evidence level: a real historical task and real model inference were run locally; the task/grader pair was calibrated on base and known fix; the two model generations did not produce executable changes. No model was trained, no capability improvement was measured, and no GPU/distributed training was verified.
