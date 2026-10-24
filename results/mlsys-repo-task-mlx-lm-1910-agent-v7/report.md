# MLX-LM #1910 real-model coding-agent feasibility

**Outcome: FAILED — ABORTED_RESOURCE during greedy model initialization.** The Development grader calibration completed successfully, but the real-model trajectory did not reach a recorded generation or patch. The sampled attempt was NOT_RUN after the abort.

The prospective task was `ml-explore/mlx-lm#1910`, assigned Development only and separated from the frozen #1904 parser-family task. The source base (`bacd9e8e1f3a5be7dd64bae3a9ad8d74471ccc3a`), fix (`a537041adc86d73cfae4cc1375b2c8851c652c60`), test, model snapshot and runner identities are hash-locked in the protocols. The task source is MIT-licensed; the locally cached `mlx-community/Qwen3-4B-Instruct-2507-4bit` model card identifies Apache-2.0. The model revision is `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Local records contain no earlier #1910 model rollout. The upstream merge date postdates the model snapshot, which is a freshness proxy and cannot prove absence from pretraining.

The task grader passed two calibration layers. Protocol v6 reached the expected upstream regression outcomes (baseline FAIL, exact fix PASS, and all three wrong mutations FAIL) and the independent semantic oracle returned baseline 0/9, exact fix 9/9, with each wrong mutant detected. The candidate grader then independently repeated baseline FAIL / exact fix PASS / three mutant FAIL before any model prompt. This qualifies the evaluator for this Development task only; no held-out task was tested.

The model process was launched with greedy settings, but swap grew from 2683.25 MiB to 3935.81 MiB at the fourth-second poll, crossing the locked 2900 MiB abort threshold. The monitor terminated it. Swap measured 4038.50 MiB immediately after termination; system free memory was 32% at the last in-run sample and recovered to 43% after cleanup. The run created no agent transcript or `agent.json`, so full model loading is unconfirmed and no response tokens were recorded. The unchanged baseline later failed the independent grader as expected. No public tests were invoked by the agent, no edits were applied, no reward was assigned, and no optimizer step was committed.

| Measure | Result |
|---|---:|
| Independently accepted model patches | 0 / 0 generated attempts |
| Unique Development task success | 0 / 1 attempted task |
| Greedy attempt | ABORTED_RESOURCE after 4.17 s |
| Sampled attempt | NOT_RUN |
| Agent valid edits / public-test calls | 0 / 0 |
| Reward variance | Not defined |
| Model generation tokens | 0 recorded |
| Optimizer steps | 0 |
| Peak observed swap | 4038.50 MiB after termination |
| Minimum in-run free-memory reading | 32% |

Historical setup failures and negative outcomes remain in the versioned v1–v7 result directories. The v7 raw summary is unchanged; `interpretation.json` classifies the attempt using the preregistered resource threshold.

Reproduction of the task workspace from a clean checkout: run `python3 scripts/prepare_mlx_lm_1910_agent_task_v1.py --task-root benchmarks/mlsys_repo_eval_v3/mlxlm_1910_agent_task`. It reconstructs and verifies local base commit `d151a804ca04296e758e77503bc552515f723399`. The frozen v7 command is recorded in `protocol.lock.json`; its complete raw run and grader records are under `run-1/`. Do not rerun inference on this machine from this protocol: the recorded abort threshold was crossed.

## Evidence level and limits

Highest achieved level: **Development-task grader calibration with mutation evidence**. A real model-backed coding trajectory was attempted but **not completed**. There is no demonstrated model-generated repair, non-degenerate reward distribution, learning signal, SFT/GRPO update, held-out result, or long-horizon result. This one prospective task cannot support a population success estimate.

## Repository validation

Three focused existing agent-contract/tool-format tests passed (36 tests total). Python byte-compilation and `git diff --check` passed. Ruff reported only two unused imports in protocol-hashed runner files (`shutil` and `sys`); all other selected lint checks passed when those frozen-source F401 findings were excluded. The repository CI result for the new PR is pending.
