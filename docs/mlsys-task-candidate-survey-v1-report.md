# MLSys coding task candidate survey v1

## Decision

**NO-GO for a new policy comparison at this gate.** The discovery pass captured
57 merged upstream pull requests across six public repositories. A deliberately
coarse metadata screen leaves 24 for manual qualification. It does not establish
15 independent, reproducible, freshness-qualified tasks, and no task has yet
passed that deeper qualification. The train/held-out split remains unfrozen and
no training or new model rollout was started.

This is a data-readiness result. It is not evidence that SFT or GRPO cannot
improve coding success. The prior source-bound readiness audit remains
reproducible and also found fewer than the required 15 eligible held-out
definitions after prior-use exclusions.

## Discovery and screen

The frozen GitHub REST snapshot and SHA-256 inventory are in
[`results/mlsys-task-candidate-survey-v1/`](../results/mlsys-task-candidate-survey-v1/).
Each candidate records its upstream PR, merge time, base/head/merge commit
SHAs, changed paths, and paths that look like tests. The API snapshot returned
57/57 requested merged PRs without request failures.

| Repository | Candidate PRs | Coarse provisional |
| --- | ---: | ---: |
| `huggingface/transformers` | 15 | 7 |
| `ml-explore/mlx-lm` | 20 | 12 |
| `ml-explore/mlx` | 11 | 2 |
| `huggingface/trl` | 6 | 1 |
| `vllm-project/vllm` | 4 | 2 |
| `ray-project/ray` | 1 | 0 |
| **Total** | **57** | **24** |

The first-pass rules retain only merged changes with a merge SHA, a changed
test path, at most 250 changed lines and eight changed files, and a merge date
strictly after 2025-08-09. It also flags title terms associated with specialized
hardware, distributed features, or broad feature additions for manual review.
The 33 exclusions overlap: 18 have no test path in the PR diff, 14 have a
specialized/feature-scope title marker, four touch more than eight files, three
exceed 250 changed lines, and two predate the freshness proxy. A candidate may
have multiple reasons.

The provisional list still includes correlated families: for example, several
MLX-LM changes concern gradient flow through MoE routing; several Transformer
changes concern checkpoint resume; and several MLX-LM tasks concern padding or
model-specific masks. The number 24 must not be read as 24 independent tasks.

## Model and resource reference

The next planned local model is
`mlx-community/Qwen3-4B-Instruct-2507-4bit` at cached revision
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The public Qwen model discussion
shows training completion on 2025-08-09; that date is only a conservative
freshness proxy, because the exact training-data cutoff is not published. A
future task split needs a blinded freshness probe as well as a post-cutoff
source date. The model card identifies the MLX conversion and Apache-2.0
license ([model card](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/tree/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b),
[Qwen model record](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/discussions/6/files)).

The current machine is a MacBook Pro M1 Pro with 32 GiB unified memory and
Metal. No paid API or external GPU was used for this metadata survey. The 24
provisional entries have not been cloned, their tests have not been run on the
M1, and per-file licensing, transitive test dependencies, isolation, task
extraction cost, and source/test leakage have not been checked. Repository
SPDX values in the lock are discovery metadata, not a derivative-work license
review.

## Required next gate

Before any no-update/SFT/GRPO rollout, manually inspect and deduplicate the 24
provisional entries, reproduce the selected tests at the pinned revisions on
the M1 or an explicitly permitted CPU path, make the hidden grader independent
of candidate-visible source, confirm per-file licenses and post-cutoff
freshness, and record prior-use exclusions. Then freeze at least 15 independent
held-out tasks plus separate development and SFT data, with source/evaluator
hashes. If fewer than 15 survive, retain this NO-GO and do not train.

Reproduce the metadata snapshot and coarse screen with:

```bash
python scripts/collect_mlsys_task_candidate_pool_v1.py \
  --output /tmp/vare-mlsys-task-survey-rerun-1/new-run
python scripts/screen_mlsys_task_candidate_pool_v1.py \
  /tmp/vare-mlsys-task-survey-rerun-1/new-run/candidate-pool.json \
  --output /tmp/vare-mlsys-task-survey-rerun-1/new-screen
python scripts/audit_mlsys_task_candidate_pool_v1.py \
  results/mlsys-task-candidate-survey-v1
```

The collection script refuses to overwrite an existing output directory and
returns nonzero if any candidate could not be captured. Keep reruns outside
the frozen evidence root: the root manifest covers only the audited run-1 and
screen-1 artifacts, and the auditor rejects extra payload files in that root.
Reruns are new metadata snapshots and are not interchangeable with this
frozen survey evidence.
