# Coding-agent evaluation

VARE includes interfaces and evidence tools for evaluating agents on repository tasks. The implemented path captures bounded agent execution, task workspaces, evaluator results, and source-linked records. This supports reproducible engineering experiments; it does not currently establish a measured coding-capability result.

## Execution and workspace binding

The [task catalog](../src/vare/environments/catalog.py) loads versioned task packs. `WorkspaceAgentRollout` creates a workspace and records its path. The [command adapter](../src/vare/environments/campaign.py) accepts a configured argument vector with workspace, prompt, and step placeholders; it does not invoke a shell. The command's child environment is allowlisted, and stdout/stderr, timeout, cancellation, and process lifetime are bounded by the owned-command helper.

Subprocess ownership is an execution-lifecycle control. Same-user commands and their descendants are not thereby sandboxed. Environment allowlists are hygiene, not isolation. Run candidate code and its grader only inside a separately administered VM or container with restricted files, network, credentials, and evaluator access. The [Docker isolation backend](../src/vare/container_isolation.py) provides an explicit backend interface; operators must configure and trust the isolation boundary.

For evidence collection, the read-only [workspace verifier](../scripts/verify_agent_workspace.py) binds a task-pinned Git HEAD, permitted workspace changes, source blobs, final source bytes, candidate diff, and grader-reported hashes. The [offline record collector](../scripts/record_coding_agent_baseline.py) checks agent/grader records against a versioned protocol, retains negative outcomes, and writes a manifest. It does not execute candidate code or attest that an independently trusted grader ran.

The collector's hashes establish byte consistency within the declared trust assumptions. They do not prove who created the records, establish a trusted execution environment, or prevent a concurrently mutating process from changing the workspace. For measurements, use an immutable workspace snapshot and independent grader provenance. Do not treat the host process runner as a hostile-code boundary.

## Problem to evidence path

| Stage | Implementation and checks | Evidence status |
| --- | --- | --- |
| Task specification and workspace | [Task spec](../src/vare/environments/spec.py), [workspace and catalog](../src/vare/environments/catalog.py), [environment tests](../tests/test_environments.py) | Implemented interfaces; individual historical task packs have their own immutable revisions and calibration records. |
| Agent process lifetime | [Campaign and command adapter](../src/vare/environments/campaign.py), [ownership regression](../tests/test_command_agent_ownership.py), [campaign tests](../tests/test_environment_campaign.py) | Bounded subprocess ownership and cancellation behavior under deterministic tests; not a security sandbox. |
| Grader execution | [Environment evaluator](../src/vare/environments/runner.py), [optional container boundary](../src/vare/container_isolation.py) | Evaluator commands must run within the configured trust boundary. The local runner alone does not isolate arbitrary candidate code. |
| Record and source binding | [Workspace verifier](../scripts/verify_agent_workspace.py), [record collector](../scripts/record_coding_agent_baseline.py), [binding tests](../tests/test_coding_agent_workspace_binding.py), [collector tests](../tests/test_record_coding_agent_baseline.py) | V2 deterministic controls validate evidence consistency against synthetic fixture code. |

## Current evidence status

The latest [workspace-bound v2 report](verified_coding_agent_baseline_v2.md) is **E0 deterministic integration evidence**. It retains regression controls for missing workspaces, altered source, forged hash fields, staged/untracked paths, symlinks, wrong HEAD, undeclared tools, and valid negative grades. The v2 protocol is premeasurement and does not retroactively change v1 records. A later live CPU run of Qwen2.5-Coder-14B-Instruct on the MLSys pack is recorded separately in [mlsys-coding-agent-e2e-v1-report.md](mlsys-coding-agent-e2e-v1-report.md): 2 of 5 consumed development tasks and 1 of 4 untouched held-out tasks passed. That count is not a success rate and does not replace the v2 controls.

No fresh live coding-agent execution, newly calibrated real-repository task cohort, independently graded task-success measurement, or agent-training outcome is reported by this implementation. A task integration or a passing synthetic regression harness is not evidence of model capability. The earlier [environment-factory overview](environment_factory.md) and [v1 report](verified_coding_agent_baseline_v1.md) retain their original scope and history.

## Reproduction

From a Python 3.11+ development environment, run the deterministic record and workspace-binding checks:

```bash
python -m pytest tests/test_record_coding_agent_baseline.py
python -m pytest tests/test_coding_agent_workspace_binding.py
python -m pytest tests/test_command_agent_ownership.py
```

For a future live run, first qualify and freeze a fresh task and grader, configure an independently administered isolation boundary, and follow the exact protocol and reproduction instructions in the [v2 report](verified_coding_agent_baseline_v2.md#reproduction). The premeasurement v2 protocol alone does not qualify an already consumed task as held-out evidence.
