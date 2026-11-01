# VARE real-model integration and consolidation — 2026-11-01

## Scope and starting state

This run began from `main` at `93ee9231e5da97b29fa0aa1745dfab332a6a952d`, with 27 open Draft PRs. The integration branch is `codex/real-model-integration-20261101`. It extracts the usable GRPO and long-horizon code from #168 and #172 into a main based change, preserves frozen protocols, and adds real Qwen execution evidence. The exact post-merge `main` SHA and GitHub PR actions are in the companion action ledger.

## Implemented code

- `src/vare/grpo_mlx.py` contains the MLX GRPO objective, completion-only loss masks, policy-age checks, zero-variance group handling, checkpoint save/load, and numerical reference path.
- `scripts/qwen3_function_repair_grpo.py` runs a pinned Qwen3-4B policy against parent-owned function contracts, executes each candidate case through the existing isolated Docker worker, and compares typed JSON outcomes in the parent (including exact `None` handling). It records source, task, optimizer, and checkpoint identities. Its optional `--protocol` locks the seed, source hashes, and task hashes for additive follow-up runs; the original protocol remains unchanged.
- `scripts/long_horizon_eval.py` connects the pinned MLX policy to the existing fixed controller and durable hash-chained trajectory store. It records protocol identity on the episode, reconstructs workspaces from committed edits on resume, and supports SIGKILL injection after a selected commit.
- `scripts/funcy_tree_oracle.py` scopes independent grading to one Train task and four parent-owned expected cases. It does not run arbitrary repository pytest in the candidate interpreter.
- `scripts/run_local_agent.py` has hardened syntax checks against symlink traversal.

The objective tests compare the on-policy gradient with a float64 reference, check the completion mask, zero-variance no-op, stale-policy rejection, checkpoint round trips, and unchanged ratios on policy. The exact focused suite passed **213 tests, 0 failures, 0 errors, 0 skips** on the final integration branch; the JUnit record is preserved with local evidence.

## Real GRPO attempts

Both experiments used `mlx-community/Qwen3-4B-Instruct-2507-4bit` at local snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, Python 3.12.12, MLX 0.32.3, MLX-LM 0.32.0, HuggingFace Hub 2.2.0, NumPy 2.5.4, and Docker image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`. The host was an M1 Pro MacBook Pro with 32 GiB unified memory; model execution used MLX/Metal and candidate scoring used Colima Docker.

| Seed | Train generations | Informative groups | Optimizer updates | Development before |
|---:|---:|---:|---:|---:|
| 20261101 | 24 | 0 of 6 | 0 | 2/4 |
| 20261102 | 24 | 0 of 6 | 0 | 2/4 |

For each seed, five task groups scored 4/4 and one task group scored 0/4, so every group had zero reward variance. The trainer skipped the update both times. No LoRA parameters changed, adapter was written or reloaded, post-update Development comparison occurred, or learner interruption recovery was claimed. This is a real-model **NO UPDATE** result, not an executed learning update. Across two seeds and 48 trusted-scored Train candidates the selected task distribution did not provide an informative group; the reward threshold was not changed and no update was forced. Development results are descriptive, not held-out evidence.

Seed 20261101 uses protocol SHA-256 `3fd79045426ec19c9eb892d3ad519aff831e863cdfb3dfaf4339af0b5533ed62`, run source commit `f77ae59bff2aeb5b7bb2e86ff8b89475db0f1127`, and runner SHA-256 `99f7f63ff96a7369b9750269097494dcbb13c38a87cf3dba589ecf59bf68d2b4`. Seed 20261102 is an additive preregistered follow-up, SHA-256 `bf7bd9aff2a7cd595355aedbb62b4ad3553599ebe3ecd3254324be288c7402ce`, at source commit `1a8c893f7311cb4218c8f4506d1a3fbf1a1fa610`, with runner SHA-256 `0e78815b930b5ed25cbcffcea5a596cebddc51d15fdfdb1d60b4d5973b2d6324`.

| Seed | Wall time | Process CPU | Peak MLX allocation | Peak RSS | Available memory at start | Free disk at start |
|---:|---:|---:|---:|---:|---:|---:|
| 20261101 | 151.26 s | 51.45 s | 2.614 GiB | 2.461 GiB | 13.81 GiB | 19.26 GiB |
| 20261102 | 150.95 s | 49.34 s | 2.614 GiB | 2.442 GiB | 14.50 GiB | 19.12 GiB |

Swap was 2.072 of 3.0 GiB at both starts and did not increase during either run. The runner recorded MLX allocation, process CPU and resident memory; GPU utilization was not sampled.

## Real long-horizon episodes and recovery

The frozen episode used the same pinned base Qwen3 policy and `full` controller on Train task `Suor__funcy.207a7810.func_basic__jtj6rhfx` from `Suor/funcy`, source revision `4895f919dfd83e24f0334421c5bac104b69d8946`, seed 0, six tool interactions, four-case isolated oracle, 512 generated tokens per turn maximum, and 120-second leases. The original recovery protocol SHA-256 is `1444cc7612568069ad129b5f5f2bb8d2765c4c5132c80ad499f0f089b543da22`; the post-edit recovery addendum SHA-256 is `a7bf2c1fc9c37167c930167e0b6008f51337ec9d79fd1141514f61aebb33bbea`. The long-horizon runner SHA-256 is `41a441a5107b43261f379e538d2a3cf19109c05cf7cc52d614ecbd828f514fd2`.

The real policy completed one uninterrupted episode and two SIGKILL recovery episodes. The first kill was after committed step 1, before an edit. The second was after committed step 3, after `replace_text` had changed `funcy/tree.py` and before syntax verification or grading. Both kill commands exited 137. After each recorded 120-second lease expired, rerunning the identical command resumed the existing episode. For the post-edit kill, SQLite showed four committed steps, `edit_applied=true`, the expected file SHA-256, and `next_step=4` before recovery.

All three completed trajectories have the same normalized 16-step semantic trace (trace SHA-256 `33f48f41ea29738abb20fd10bbe916a69066861165f8404d3f1c78ed39678a8c`), the same six operation IDs and tool calls, and the same final file SHA-256 `afe7e2e5bfb2a9998e21b79eac09b37469cdf7a7b058e835af3cd2526ccfbed2`. Each trajectory store passes hash-chain audit with zero effective duplicates. Semantic comparison excludes only timing values, per-call seconds, and observed peak memory; model outputs, arguments, tool results, state hashes, stop reason, and operation IDs remain compared.

The action sequence was `list_files`, `read_file`, `search_text`, `replace_text`, `check_syntax`, `finish`. Each episode used 9,939 prompt tokens and generated 192 tokens. On the uninterrupted episode, measured model generation took 31.196 seconds, tools 0.063 seconds, and trusted grading 1.411 seconds; peak policy memory was about 3.20 GiB. Model generation was 95.5% of combined generation, tool, and grading time for this one episode. This is evidence that inference/prefill dominates this measured path; it is not a general workload estimate or speedup claim.

The trusted oracle passed the pinned reference 4/4 and a trivial `None` candidate 0/4. The Qwen candidate passed `scalar_root`, `false_and_none_leaves`, and `empty_tree`, but failed `nested_lists`; the final trusted outcome is `case_failed`. It changed only the target file and passed syntax checking. There were no public tests and no independently verified task success. This episode demonstrates real multi-turn execution, durable state, trusted scoped grading, and recovery—not coding capability improvement.

## R7 and historical evidence boundary

The immutable Run-002 R7 report records 903 attempts: 103 valid reference outcomes, 769 `reference_tests_missing`, and 31 `reference_fails`; 12 repositories were admitted, 19 had insufficient valid tasks, and 4 had environment-install failures. Its fixed 26 × 8 qualification gate still refused the held-out pool. The retained report stores per-task counts and stdout hashes, but not the stdout or JUnit XML required to separate collection/import errors from absent requested node IDs. I did not rerun the 903 attempts or reuse the held-out tasks. The six public Run-003 fixtures validate taxonomy behavior only; they are not evidence about the historical Run-002 failure distribution.

## Limits and unresolved work

- No GRPO update, adapter, post-update Development measurement, or learner checkpoint recovery was achieved because both preregistered seeds had zero informative groups.
- The long-horizon trial covers one exposed Train task and one deterministic seed. The trusted grader is only the declared four-case function contract; arbitrary repository pytest grading remains unsafe in the same-interpreter design documented in `docs/learning-pipeline-grader-isolation.md`.
- No held-out pass-rate claim or statistical improvement estimate is supported.
- Inference dominates the measured episode, but this one-task sample did not establish a safe optimization contract. The previous edit-format Qwen proposals remain rejected; no speedup is claimed.
- Run-002 failure diagnosis remains blocked by missing retained pytest diagnostics and held-out reuse constraints.

## Reproduction commands

From the repository root, set `RUN_DIR` to an empty evidence directory. Run model phases serially and keep the model snapshot local/offline.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DOCKER_HOST=unix:///Users/<user>/.colima/default/docker.sock \
  .venv/bin/python scripts/qwen3_function_repair_grpo.py train \
  --run-dir "$RUN_DIR/grpo-seed-20261102" --seed 20261102 \
  --protocol protocols/qwen3_function_repair_grpo_dev_v1_seed_20261102_addendum.json \
  --sigkill-after-update
```

The actual run returned 0 through the no-informative-group path; therefore the registered update-boundary fault was not triggered. The recovery command is applicable only after a real update and durable `update-0001` checkpoint.

The post-edit long-horizon recovery was run with the following arguments. The same command was rerun after the lease expired to resume the episode.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DOCKER_HOST=unix:///Users/<user>/.colima/default/docker.sock \
  .venv/bin/python scripts/long_horizon_eval.py run --arm base --policy-version 0 \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --revision 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  --controller full --manifest configs/learning_pipeline_v1/train.manifest.jsonl \
  --cache "$HOME/.cache/vare-learning-pipeline-v1" --limit 1 --seeds 1 \
  --output "$RUN_DIR/post-edit-recovery" \
  --instances Suor__funcy.207a7810.func_basic__jtj6rhfx --lease-seconds 120 \
  --protocol protocols/qwen3_long_horizon_train_v1_post_edit_recovery_addendum.json \
  --grader-factory scripts.funcy_tree_oracle:grader_factory --crash-after-commit-index 3
```

The machine readable summary and normalized recovery comparison are in `results/real-model-integration-20261101/`; complete local model outputs and hash manifest are linked from the accompanying deliverable.
