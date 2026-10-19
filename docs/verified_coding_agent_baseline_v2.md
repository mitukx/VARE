# Verified coding-agent baseline v2 — workspace-bound audit

**Evidence status: E0 deterministic integration controls only.** The v1 implementation, v1 premeasurement protocol, and existing historical tasks/measurements are preserved. V2 corrects a concrete consistency gap: two records containing matching hashes do not establish that the final candidate workspace actually contained those bytes.

## Bounded change

- `scripts/verify_agent_workspace.py` is a **read-only** Git/object inspection helper. It does not import candidate modules or execute their code. It verifies: an owned `.git` directory; task-pinned HEAD; cleanly allowlisted unstaged changes only; no staged/untracked/protected changes; regular non-symlink task files; SHA-256 of pinned base blobs and on-disk final source files; exact `git diff --binary` correspondence with the agent transcript; and correspondence between the actual final source and grader's reported source/diff hashes. Git external diff filters, text conversion, fsmonitor and lazy object fetching are disabled.
- `scripts/record_coding_agent_baseline.py` adds `--workspace` and requires it when `workspace_verification.required` is true in a protocol. It checks the copied snapshot of evidence inputs (rather than copying inputs then independently re-reading mutable originals). It rejects source diffs without any successful recorded `apply_patch` tool, and rejects undeclared tools when an allowlist is in the protocol. It retains negative grader outcomes as negative evidence.
- `protocols/verified_coding_agent_baseline_v2.json` is a **new**, premeasurement protocol; v1 is not retroactively altered. It uses the same previously consumed RVL historical integration control and does not qualify as held-out evaluation.
- `tests/test_coding_agent_workspace_binding.py` exercises a *real temporary Git repository* containing **synthetic fixture code**, not third-party code execution. Controls include valid binding, negative grading, missing workspace, staged/untracked files, forged source/diff fields, out-of-band edits, symlinks, wrong HEAD, unknown tools, and unauthorized write surfaces. It is E0 only.

## Reproduction

From a clean checkout that includes v1 and v2 changes, verify the protocol snapshot hashes, then run:

```bash
python3 -m pytest -q tests/test_record_coding_agent_baseline.py
python3 -m pytest -q tests/test_coding_agent_workspace_binding.py
```

With a previously prepared pinned workspace, a retained agent JSON file and an independently generated grader result, invoke:

```bash
python3 scripts/record_coding_agent_baseline.py \
  --protocol protocols/verified_coding_agent_baseline_v2.json \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --workspace /isolated/agent-101 \
  --agent-record /isolated/out/agent-101.json \
  --grader-record /isolated/out/grade-101.json \
  --output-dir results/verified-coding-agent-baseline-v2/seed-101
```

The output directory must not exist before invocation. It retains input record snapshots, JSON summary, SHA-256 manifest, and the workspace audit diagnostics. The collector exits `0` for a structurally valid negative grade, `1` for invalid evidence, and does **not** run any candidate code. If a grader times out without emitting a result JSON, retain the failed attempt with invalid evidence instead of silently excluding the seed.

**Important:** Run `scripts/grade_task.py` only inside a separately configured filesystem/network-isolated VM/container with no candidate-accessible secrets or trusted evaluator files. The original runner and grader do not constitute such a sandbox. Do not use this read-only audit to justify executing arbitrary model-generated code on the host.

## Claim boundary and remaining gates

The v2 tests validate source/record consistency, not task capability. SHA-256 hashes do not authenticate who created the inputs. A hostile process that fabricates agent, grader and workspace jointly may fool a consistency-only auditor; real independent evaluator attestation and sandboxing remain separate requirements. Concurrent workspace mutation is not addressed by snapshot isolation; retain an immutable workspace image for final measurements.

No newly calibrated historical software-engineering task, model/tool rollout, independently graded live execution, optimizer update, held-out improvement, or hours-long trajectory is claimed. On the available container, GitHub clone/network access and an isolated model/grader runtime were not available. Next experimental gate: **one pinned, isolated real agent run plus a fresh independently calibrated task cohort**, under a separately frozen and reviewed protocol.
