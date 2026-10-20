# Actual-model MLSys update study: readiness NO-GO

**Decision: NO-GO for the held-out no-update/SFT/GRPO comparison.** The
Apple-Silicon feasibility work completed real MLX inference, a ten-step LoRA
update, independent coding-agent rollouts, and a bounded trajectory-recovery
comparison. It did not measure held-out task success before and after an
update. The frozen readiness audit finds zero eligible fresh held-out tasks;
this is an execution-readiness result, not evidence that an update cannot
improve task success.

## Frozen decision rule and task inventory

The readiness lock is
[`mlsys_actual_model_update_no_go_v1.lock.json`](../protocols/mlsys_actual_model_update_no_go_v1.lock.json).
It requires at least 15 unique, independently graded held-out task definitions
before training or final evaluation, after excluding tasks already used by PR
#84 through #90. The inventory pins each pack commit, manifest, split, task ID,
task JSON hash, and exclusion reason.

The audit covers 19 task entries across packs v1–v3, representing 14 unique
task definitions after deduplicating v3 references to v1 tasks. All nine v1
tasks and all three v2 tasks are from PR #84 or #86 and are excluded by the
no-reuse rule. The five v3 train tasks reuse v1 task definitions. The two new
v3 tasks, EMA bias correction and label smoothing, were both used in the PR #90
matched exploratory cohort. The remaining GQA candidate is unassigned and its
mutation condition is not represented in the separated-call schema. Therefore
the eligible final held-out count is **0/15**.

The read-only audit checks the three pack commits, manifest and split hashes,
task-file inventory, IDs, task types, split membership, provenance fields, and
task JSON hashes. It does not load a model or open hidden grader files.

```bash
python scripts/audit_mlsys_update_readiness_v1.py \
  --protocol protocols/mlsys_actual_model_update_no_go_v1.lock.json \
  --output results/mlsys-actual-model-update-readiness-v1/audit.json
```

The retained audit reports 19 task entries, 14 unique definitions, zero
eligible tasks, and `NO_GO`; all source and inventory checks pass. A separate
test edits a copied task definition and verifies that the hash audit fails.

## Feasibility measurements retained before the NO-GO

- **Hardware:** MacBook Pro M1 Pro, 32 GiB unified memory, macOS 27.0.1, Python
  3.12.12, MLX/MLX-LM 0.32.2. PyTorch MPS was available, but the MLX runs used
  MLX's selected Metal GPU device. Swap was already high at roughly 7.4 GiB
  used before the runs and stayed unchanged.
- **Inference:** Qwen3-4B-Instruct-2507-4bit, three 96-token greedy generations
  per backend. Median decode was 55.67 tokens/s on Metal and 55.79 tokens/s on
  MLX CPU, with identical output hashes. Metal peak allocation was 2.41 GB and
  process max RSS was 2.65 GB. This short batch-one measurement showed no
  throughput advantage for Metal.
- **Actual update:** Qwen2.5-0.5B-Instruct, locally converted to 4-bit MLX
  weights; one previously grader-verified development trajectory; LoRA rank 8,
  batch size 1, sequence limit 1024, Adam at `1e-5`, and ten optimizer steps.
  Training loss on that example fell from 1.281 to 0.807. The adapter has
  366,592 trainable parameters, including 202,752 nonzero LoRA-B parameters.
  Update time was 13.71 s; peak MLX allocation 1.80 GB and process max RSS
  0.92 GB. A second audit parsed the MLX progress log and reloaded the adapter
  on Metal. No optimizer-state resume, no-update comparison, GRPO, or held-out
  task evaluation was performed.
- **Coding-agent rollouts:** the new matched exploratory cohort recorded 4B
  Qwen3 at 1/2 verified tasks and 7B Qwen2.5-Coder at 0/2. The 4B task that
  passed used 516 generated tokens and 27.4 s. The different model families,
  two tasks, and one greedy sample per task do not support a model-size ranking
  or broad success estimate. The earlier 4B development cohort recorded 2/5
  after a disclosed post-run grader correction; it is not confirmation data.
- **Recovery:** three repetitions per arm and condition on one already exposed
  development task. After worker SIGKILL between a successful public-test tool
  operation and the next committed step, restart-from-zero used four tool
  executions with two duplicates; VARE resume used three executions with one
  duplicate. Both recovered a passing source state in 3/3 runs. Median time was
  0.664 s for restart and 3.772 s for VARE, because the three-second lease
  expiry dominated. This is a narrow replay-versus-latency tradeoff, not a
  broad reliability gain.

## Evidence bundle and reproduction

The adapter is larger than the repository's 1 MiB tracked-blob limit. The
separate evidence bundle contains the adapter, raw summaries, the original
training log, model/file hashes, and audits. Bundle SHA-256:
`b3c4911ef5b0775353d11d96e0ee4ba577044d6b8859f5e51aec6fc0bacd1725`.

Download both `evidence_bundle.tar.gz` and `evidence_bundle.tar.gz.sha256` from the
[Apple Silicon update evidence release](https://github.com/mitukx/VARE/releases/tag/vare-apple-silicon-update-v1-evidence),
verify the archive, and extract it. From the VARE checkout:

```bash
shasum -a 256 -c evidence_bundle.tar.gz.sha256
tar -xzf evidence_bundle.tar.gz
python scripts/audit_mlx_lora_update_smoke_v1.py \
  evidence/results/mlx-lora-update-smoke-v1/run-1 \
  --adapter-checkpoint \
  evidence/results/mlx-lora-update-smoke-v1/run-1/adapter/adapters.safetensors
```

The independent update audit reproduced `PASS_UPDATE_FEASIBILITY_ONLY` and
verified the adapter SHA-256, Metal selection, ten logged optimizer steps,
finite decreasing training loss, nonzero adapter weights, reload generation,
and resource limits. The original runner summary remains `non_pass` because
its step parser did not recognize the MLX-LM 0.32.2 progress table; both the
raw summary and the correcting audit are retained.

## Decision and claim boundary

No SFT-versus-no-update or GRPO-versus-no-update comparison was started. No
newly frozen 15–20-task confirmation cohort exists, and the 0.5B update smoke
does not have a matching base-model task-success gate. The available 4B/7B
exploratory cohort is already consumed and too small for that gate. The next
work is task authoring and independent grader review until the frozen readiness
gate can pass; training and final held-out evaluation remain stopped meanwhile.

This NO-GO is falsifiable by adding a versioned pack of at least 30 independently
screened candidates, freezing 15–20 untouched held-out tasks and passing the
development-only base feasibility gate. It makes no claim about the eventual
effect of SFT or GRPO.
