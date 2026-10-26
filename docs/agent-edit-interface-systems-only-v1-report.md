# Deterministic coding-interface replay and recovery: systems-only result

**Decision:** narrow systems smoke test passed; model-learning and coding-capability gates remain **NO-GO**.

## Scope

The frozen protocol [`agent_edit_interface_systems_only_v1.json`](../protocols/agent_edit_interface_systems_only_v1.json) (SHA-256 `75e2e4c7e87a0bfefd0c4795f36adf9862470fdcc926359d32069f1d82e79fc6`) compares synchronous one-worker and asynchronous two-worker execution of deterministic known-fix controllers on two MLX-LM Development families: #2004 and #1996. Each task is exercised through the lease-bound line-edit interface and the legacy structured-replace interface. The controller is given the upstream fix; these are not agent-generated repairs.

The model-coupled Qwen3-4B pilot stopped before generation. Three wrapper setup failures were retained under [`model-run-v1`](../results/agent-edit-interface-v1/model-run-v1/), including the missing `psutil` dependency, a process-tree RSS key mismatch, and an `inference_budget` key error. There are no model outputs, edits, grades, optimizer steps, or model capability observations from that pilot. The frozen protocol prohibited further same-protocol calls, and none were made.

## Measured outcomes

The deterministic replay completed all **8/8** task × interface × execution-mode cells. Every workspace contained the exact pinned fix bytes and passed the independent, network-denied grader. A separate injected process kill occurred after the atomic edit and fsynced journal but before grading. The child exited with SIGKILL (`-9`); recovery found the exact expected target hash, graded it successfully, and made no duplicate edit.

| Measure | Synchronous, 1 worker | Asynchronous, 2 workers |
|---|---:|---:|
| Verified cells | 4/4 | 4/4 |
| Execution plus grading wall time after workspace materialization | 5.3363 s | 2.9363 s |
| Queue-inclusive cell latencies | 1.4905, 2.7927, 4.0689, 5.3355 s | 1.4574, 1.4575, 2.9349, 2.9350 s |

The paired synchronous/asynchronous wall-time ratio is **1.817** for this one four-cell workload on one Mac. This is descriptive throughput only: it is not a replicated async speedup estimate, and it measures neither RL rollouts nor model quality. Using the pinned Qwen tokenizer without loading weights, the serialized known-patch tool requests used 878 tokens for lease-bound edits (718 + 160) and 1,155 for the legacy structured-replace calls (918 + 237), a 23.98% difference for these exact requests. These are not generated-model tokens.

Across 25 resource samples, minimum available memory was 9,586,982,912 bytes, swap growth was 0, minimum free swap was 1,809,383,424 bytes, minimum free disk was 116,108,607,488 bytes, and peak process-tree RSS was 457,457,664 bytes. No resource abort occurred. The host/process-tree sampler was used; macOS cgroup telemetry is unavailable.

## Evidence integrity and correction

The first runner invocation failed before protocol validation, workspace creation, or grader/model calls because direct-file execution did not put the repository root on `sys.path`. That failed launch is preserved in [`systems-only-v1-launch-failure.json`](../results/agent-edit-interface-v1/systems-only-v1-launch-failure.json). The same committed runner and frozen protocol then completed through module invocation; neither was changed for that successful run.

The frozen traces contain **two identical `known_fix_controller_complete` journal events per cell**. The independent audit confirms valid journal hash chains and exactly one committed edit per operation; the duplicated event is a logging defect, not an additional edit. The runner correction and regression test remove the duplicate for future runs. Historical traces remain unchanged and are reported with a warning in [`independent-audit-v2.json`](../results/agent-edit-interface-v1/systems-only-v1/independent-audit-v2.json). This second audit checks source and supporting files from the protocol-pinned commit, verifies the previous artifact manifest and every entry, and writes a separate versioned audit and manifest without overwriting the original audit outputs.

The top-level evidence index [`systems-only-v1-evidence-manifest.json`](../results/agent-edit-interface-v1/systems-only-v1-evidence-manifest.json) binds the frozen protocol, model setup-failure manifest, launcher failure record, raw successful-run manifest, and versioned independent audit. The raw run includes per-cell records, tool traces, journals, trusted grader logs, resource samples, and recovery data.

## Research interpretation

- **Task certification:** two Development task families were used; no held-out task was evaluated.
- **Systems reliability:** the deterministic edit/recovery boundary passed in this bounded smoke test, with the duplicate journal-terminal marker warning above.
- **Model capability:** no model emitted an output in the frozen model pilot.
- **Learning:** optimizer steps = **0**; no SFT, GRPO, policy update, model-generated shortcut, or held-out comparison occurred.
- **Verifier-contamination hypothesis:** this replay does not measure it. The existing false-reward controller evidence is still from one selected #1912 reward table. Replaying that same three-action table through a trainable controller would not provide a distinct task-level test, so no controller training was started.

The inspected PR #123 evidence-chain repair is already present at head `a02d6cb64f2531cbe6152266b431d8a466f0bdbc`: its auditor reports 41 artifacts, preserves the historical manifest mismatch, reconstructs all 36 outer entries, and binds all 18 scored outcomes. Exact-head CI passed and the independent review summary reports no findings. PR #123 remains unmerged; this report does not change it.

## Reproduction and hashes

Run the independent systems audit from the repository root with:

```bash
python -m scripts.audit_agent_edit_interface_systems_only_v1 \
  --results results/agent-edit-interface-v1/systems-only-v1 \
  --source-repo source/mlx-lm.git \
  --worktree-root ../coding-agent-interface-v2/work/agent-edit-interface-systems-only-v1 \
  --protocol protocols/agent_edit_interface_systems_only_v1.json \
  --audit-id v2
```

The audit reports 8 verified cells, status `PASS_WITH_JOURNAL_WARNING`, eight repeated-marker warnings, and a descriptive async/sync ratio of 1.817355. The successful run itself is not rerun by the audit. Its aggregate JSON SHA-256 is `5a63f9ec901d3204c42f1fa54309bcca1fc8f6dd3167f09d5469831efd89133b`; the v2 artifact-manifest SHA-256 is recorded beside that manifest and is `17dbb7402de20743d100177ea0f61af2d8917d2c6db56033f49221942ea08908`.
