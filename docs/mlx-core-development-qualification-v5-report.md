# MLX core Development task qualification v5

**Decision: GO for two source/grader-certified Development task families. NO-GO for held-out model learning.** The run reproduced the pinned MLX CPU baseline, accepted both exact upstream repairs, and rejected both frozen incorrect mutants. It did not run a candidate model, train a policy, or evaluate held-out tasks.

## Outcomes

| Task family | Baseline | Exact fix | Incorrect mutant | Decision |
| --- | --- | --- | --- | --- |
| MLX #4729, negative scalar index in `mx.take` | Rejected | Accepted | Rejected | Certified, Development only |
| MLX #4730, `mx.random.normal` shape expansion | Rejected | Accepted | Rejected | Certified, Development only |

Each decision combines the pinned upstream regression test with independent semantic checks on CPU results, shape/broadcast behavior, and expected exceptions. The baseline and mutant phases rejected both task families; the exact-fix phase passed both. The exact candidate sources, public test files, outputs, commands, resource samples, and phase records are in the [raw run](../results/mlx-core-development-qualification-v1/run-5/). The independent auditor verified all 32 manifest entries and reconstructed `PASS_TWO_DEVELOPMENT_SOURCE_GRADER_CERTIFICATES` in [its report](../results/mlx-core-development-qualification-v1/run-5-independent-audit.json).

The run is bound to protocol SHA-256 `1fee3fd17fbda00a030cd87c9364e56297e09acecfb39c3da76ab8cba877d8f6`; the output manifest is `d5c058a0a2b798583696fe7e7d9a420a00cc7aba4663f6e0f3d4493faf40a175`, the run summary is `144383ac1d8eb1f4a7a2855f4522aa02462b261ad76466102ce8900d54fb4e73`, and the independent audit is `81849c340276d7a2d09511a7853ab5c0d81efd4243fda9f62d23e41823978574`.

## Dependency closure and resources

V4 remains a frozen NO-GO: its offline CMake configuration failed before compilation because `gguflib` and nanobind's `ext/robin_map` were absent. V5 retained that failure and pinned complete local sources: `fmt` commit `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f`, `gguflib` commit `8fa6eb65236618e28fd7710a0fba565f7faa1848`, `nlohmann/json` 3.11.3 archive SHA-256 `d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d`, nanobind commit `68480a9e6883bb1bf65e1925423321e2a89d07f9`, and `robin_map` submodule commit `4ec1bf19c6a96125ea22062f38c2cf5b958e448e`. CMake used explicit `FETCHCONTENT_SOURCE_DIR_*` mappings with fully disconnected fetching; the build and test child processes ran with network denied.

On macOS 27.0.1 arm64, Python 3.12.12, Apple clang 17, and CMake 3.31.6, phase build times were 162.094 seconds (baseline), 12.876 seconds (fix), and 12.859 seconds (mutant). The maximum sampled process-tree RSS was 1,985,921,024 bytes during baseline compilation. The lowest available-memory sample was 8,877,686,784 bytes; free swap stayed at 1,809,383,424 bytes, with zero observed swap growth. The complete serialized build, observation, and six public-test calls totaled 192.523 measured seconds. No cgroup reading is claimed on macOS.

## Interpretation and limits

These two tasks are **Development** tasks. The operator had already inspected their upstream fixes and regression tests, and the candidate model received no prompt. They add zero families to the held-out inventory: the reported held-out source/grader count remains 10 of the unchanged 15-family gate. External model pretraining and task recognition remain unknown. The run establishes task-source and grader behavior under one locally reproducible native build; it does not establish agent success, model eligibility, coding capability, policy learning, or asynchronous reliability.

V4's failed build and audit are retained in [run 4](../results/mlx-core-development-qualification-v1/run-4/) and [its independent audit](../results/mlx-core-development-qualification-v1/run-4-independent-audit.json). V5 passed on the same host with a complete pinned dependency closure. A second-machine replay has not been performed; portability is limited to the committed source protocol, dependency identities, and reproduction commands below.

## Reproduction

For a live replay, start from the premeasurement commit `4d9ff74b2b449ac4e553c22124043c1ae50056a5`, before the V5 run directory was added. Use a full upstream Git clone so the pinned source and grader commits are available locally. Recreate the dependency tree at the paths named in the [V5 protocol](../protocols/mlx_core_development_qualification_v5.json), checking out the commits and submodule listed there. Download the JSON archive named in the protocol and verify its SHA-256 before extraction. Recreate the hash-locked build Python from `requirements/mlx-core-development-qualification-v1.lock.txt`. The run command is:

```sh
python scripts/run_mlx_core_development_qualification_v5_launcher.py \
  --source-repo source/mlx.git \
  --dependencies work/mlx-core-development-qualification-v1/dependencies-complete-v1 \
  --build-python work/mlx-core-development-qualification-v1/venv/bin/python \
  --output results/mlx-core-development-qualification-v1/run-5 \
  --worktree work/mlx-core-development-qualification-v1/scratch/source-v5 \
  --protocol-sha256 1fee3fd17fbda00a030cd87c9364e56297e09acecfb39c3da76ab8cba877d8f6
```

The frozen paths intentionally make that exact output non-repeatable in place. For an audit of retained evidence, choose a new audit-output path:

```sh
python scripts/audit_mlx_core_development_qualification_v5_launcher.py \
  --run results/mlx-core-development-qualification-v1/run-5 \
  --source-repo source/mlx.git \
  --dependencies work/mlx-core-development-qualification-v1/dependencies-complete-v1 \
  --build-python work/mlx-core-development-qualification-v1/audit-venv/bin/python \
  --worktree work/mlx-core-development-qualification-v1/scratch/source-v5 \
  --protocol-sha256 1fee3fd17fbda00a030cd87c9364e56297e09acecfb39c3da76ab8cba877d8f6 \
  --output results/mlx-core-development-qualification-v1/run-5-independent-audit-replay.json
```
