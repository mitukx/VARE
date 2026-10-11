# Catalog task-identity review

**Review type:** Independent AI-agent code review, not human review. The reviewer did not edit shared files and did not run or rerun any model.

## Target and reviewed state

- Worktree: `${VARE_CHECKOUT}`
- Branch: `codex/real-model-capability-eval`
- Git HEAD/base commit: `b3204926b3cae291f7c13e3134444c71a3a63172`
- The task-catalog changes were uncommitted in the shared worktree during review.
- `src/vare/environments/catalog.py` SHA-256: `e9b7bb8a801b96c5e8383a53904e26580125d043d651d905771e18a9ba24ccec` (unchanged across both review passes).
- `tests/test_environment_catalog.py` SHA-256 during the first pass: `07d3c0c53deecb3b712dc3bb1ffba9f7157aac76b7c224884a90123833f04e6c`.
- After the parent added an explicit verifier assertion for conflicting `attempt.task.metadata`, the reviewed final test file SHA-256 was `f45e196e508743e5bfcec778239b4ca80978476a84bfe987b6f115cfe235dd4f`.
- Existing baseline reproducer log `results/urllib3-3352-evaluation-v1/task-identity-baseline-tests.log` SHA-256: `fc7445f1d8abdcc912a8801f749a0144a114bc0a724eea3ed981f1f4a3fbc326`.

## Findings

The original defect was real: the base version selected the rollout catalog entry with `str(task.metadata.get("engineering_task_id", task.id))` and the verifier entry with `str(attempt.metadata.get("engineering_task_id", attempt.task.id))`. Thus metadata could make task A use task B's prompt/workspace at rollout, or B's grader at verification. The pinned-base test log shows both new regressions failed with `DID NOT RAISE`.

The revised rollout now uses `task.id` as the catalog selector and rejects a conflicting `task.metadata["engineering_task_id"]` before creating a workspace or calling the agent. The revised verifier uses `attempt.task.id`, rejects conflicts in both `attempt.task.metadata["engineering_task_id"]` and `attempt.metadata["engineering_task_id"]`, and obtains the evaluator specification from the catalog entry. The metadata key `task_spec_path` does not select the grader. I found no remaining route by which those conflicting metadata values select a different prompt or grader in the intended catalog-dispatch flow.

The second-pass test addition is valid and isolates the verifier's task-level claim: it replaces `attempt.task.metadata` with the alternate ID while leaving `attempt.metadata["engineering_task_id"]` equal to the canonical task ID. The three metadata-conflict branches now each have regression coverage: rollout task claim, verifier task claim, and verifier attempt claim.

### API boundary and counterexamples

This is a consistency guard, not authentication of the dispatch. `Task.id` itself remains the selector, and the public rollout/verifier methods accept caller-constructed `Task` and `Attempt` objects. I reproduced that a caller supplying `Task(id="always-pass", metadata={"engineering_task_id": "always-pass"})` is accepted and rollout uses the catalog's `always-pass` prompt and creates the `...-always-pass` workspace. In the intended API contract, this is expected when a trusted caller deliberately chooses a catalog task. If the system must establish that a trusted dispatcher originally assigned task A even when a caller can replace the primary ID with B, these objects carry no sealed dispatch token or separately supplied expected ID.

`Attempt.metadata["workspace_path"]` remains the verifier's candidate-workspace input. I redirected an otherwise canonical task-A attempt to a separately prepared task-A candidate directory; the task-A evaluator ran there and passed its 2/2 smoke tests. This demonstrates that a caller controlling the Attempt can select which workspace is graded. It does **not** show that the grader was changed: the verifier still used task A's catalog evaluator, and the supplied directory contained a passing task-A candidate. In the ordinary campaign path, the rollout owns and returns the Attempt and writes this path; the agent interface receives only prompt, workspace path, and step, not the Attempt object. Treat this as an API trust boundary / provenance limitation, not evidence of a remaining task-ID metadata override.

The workspace-redirection probe used an ephemeral task-spec copy with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because unrelated installed Hydra/ANTLR pytest plugins on this host otherwise fail during collection. The candidate workspace and spec were temporary; no repository source or evaluator test was modified. This was a local boundary probe, not a model run.

## Commands and outcomes

Focused current-worktree test command:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src /opt/homebrew/Caskroom/miniforge/base/bin/python3.12 -m pytest -vv tests/test_environment_catalog.py
```

Outcome after the second-pass assertion: **3 passed in 0.48s**. The first pass, before that extra assertion, also returned 3 passed in 0.47s.

The recorded pinned-base reproduction is `results/urllib3-3352-evaluation-v1/task-identity-baseline-tests.log`; it reports both identity regressions failing because the expected `ValueError` was not raised. The focused direct probes reported:

- Consistent alternate primary ID `always-pass`: accepted and selected the alternate prompt and workspace, as expected for this selector-based API.
- Canonical task-A ID/claims with caller-selected task-A passing workspace: verifier passed, with grader task ID `smoke-stable-logsumexp` and 2/2 tests passing.

## Scope and conclusion

For `EnvironmentCampaignRunner`'s catalog-produced tasks and rollout-produced attempts, the metadata-based A-to-B task selection defect is closed by the current fix, with before/after regression evidence. The change does not authenticate primary IDs or workspace paths against a separate immutable dispatch record, and should not be described as making arbitrary in-process `Attempt` objects tamper-proof. No model was run in this review, and no human review was performed.
