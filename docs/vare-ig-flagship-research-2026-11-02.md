# VARE-IG: Informative Rewards, Real Qwen Updates, Recovery, and Long-Horizon Transfer

**Research report, 2026-11-02**
**Repository:** [mitukx/VARE](https://github.com/mitukx/VARE)
**Starting `main`:** `bc8803264e4a08721b0631ec04173f2785daa676`
**Experiment model:** `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`
**Run protocol family:** VARE-IG V3b; original SHA values and portable-copy hashes are in [the protocol redaction manifest](../research/vare_ig_flagship/evidence/protocol-redaction-manifest.json).

## Summary

This milestone moved VARE past the prior zero-update result. A prospective, synthetic coding task pack produced naturally mixed rewards; the MLX trainer applied real GRPO updates to Qwen weights; durable recovery replayed and reconciled updates exactly; and a two-arm long-horizon cohort compared the base policy with a real two-update adapter.

The principal experimental result is neutral: the family-level, cost-aware scheduler found **3 informative groups in 18** in each arm, the same count as uniform sampling. It used slightly more generated tokens and measured generation-plus-verification time per informative group. The resulting policies remained at **2/4** on the small Development set, and both policies achieved **0/4** long-horizon successes when success requires both public checks and the trusted final grade. The promotion gate rejected all four runs. These outcomes do not support a capability-improvement claim.

The engineering result is stronger: a real model completion was linked to parent-verified rewards, exact behavior log probabilities, trajectory provenance, optimizer state, and a durable adapter. Both an after-publication acknowledgment failure and a pending-update SIGKILL boundary were recovered or replayed without a duplicate effective update. The project now has auditable evidence for the entire path, including hash-checked source and full long-horizon event exports.

## 1. Scope and evidence boundaries

The new VARE-IG pack has 26 fixed function tasks: 12 Train, 10 Exploration, and 4 Development tasks across parsing, graph, stream, and configuration families. These are **synthetic contracts** for studying reward informativeness and training mechanics. They are not evaluations of real open-source repository semantics. Each task has parent-owned expected results, and the committed oracle audit reports 26/26 references passing and 26/26 incorrect mutants rejected. Candidate code is run in network-disabled Docker isolation. The verifier—not candidate code—owns expected values and the authoritative verdict. This audit applies to these fixed function contracts, not to arbitrary-repository grading security.

The raw exploration rollouts, training outputs and sample records, scheduler selections, update and recovery records, Development records, and full long-horizon event ledgers are under [the evidence bundle](../research/vare_ig_flagship/evidence/). Large token-ID records and full traces are gzip-compressed; the manifest records both the stored-file and uncompressed content hashes. The [machine-readable summary](../research/vare_ig_flagship/summary.json) contains per-task and per-seed outcomes. Adapter/optimizer weights and SQLite databases remain local; their names, sizes, and hashes are listed in [local-only-artifacts.json](../research/vare_ig_flagship/evidence/local-only-artifacts.json). Protocol JSON copies replace machine-specific roots with `${VARE_ROOT}` and `${DOCKER_HOST}`; the manifest records both the original locked hash and the portable copy hash. The copies are audit artifacts and need their placeholders resolved before execution.

## 2. Prior study diagnosis and prospective task landscape

The preceding #179 Qwen study produced 48 completions over 12 groups and no informative groups. Five of six Train tasks per seed were 4/4, while the remaining task was 0/4. The saved evidence therefore supports reward saturation on that small task set; it does not establish that the verifier was wrong or that real-model GRPO cannot update.

The new 40-completion Exploration pilot used ten previously frozen Exploration contracts, four completions per task, and the base Qwen policy. Regrading used the exact saved responses—there was no resampling. The result was 15/40 passes, 34 unique source candidates, and 3/10 informative groups. Task-level outcomes were:

| Family | Task | Passes | Group | Informative? |
|---|---|---:|---|---|
| Parsing | `parse_csv_header` | 0/4 | 0,0,0,0 | No |
| Parsing | `ini_overrides` | 0/4 | 0,0,0,0 | No |
| Graph | `graph_course_order` | 4/4 | 1,1,1,1 | No |
| Graph | `shortest_unweighted_path` | 1/4 | 0,0,0,1 | Yes |
| Stream | `merge_event_ranges` | 3/4 | 1,1,1,0 | Yes |
| Stream | `running_event_counts` | 0/4 | 0,0,0,0 | No |
| Stream | `dedupe_event_ids` | 4/4 | 1,1,1,1 | No |
| Configuration | `deep_defaults_config` | 3/4 | 1,1,1,0 | Yes |
| Configuration | `duration_ms` | 0/4 | 0,0,0,0 | No |
| Configuration | `relative_path` | 0/4 | 0,0,0,0 | No |

The pilot generated 7,539 completion tokens in 210.263 seconds and spent 26.193 measured seconds in verification. Five verifier case calls were unmeasured because one output hit the 512-token cap and exited unexpectedly; that completion remained a zero reward. No unmeasured call was dropped from the accounting.

For group size four, the independent Bernoulli approximation `1 - p^4 - (1-p)^4` gives a diagnostic expectation for informative groups. At family level, the pilot observed configuration 3/12 passes and 1/3 informative groups; graph 5/8 and 1/2; parsing 0/8 and 0/2; stream 7/12 and 1/3. The corresponding plug-in IID predictions were 0.680, 0.828, 0.000, and 0.854. Observed group rates were lower for three families. Completion outcomes within a prompt are correlated, and these small family samples make the IID figures diagnostic only. The family Wilson intervals and posterior values used by the scheduler are retained in the exploration summary.

## 3. Fixed-budget scheduler comparison

The adaptive scheduler used a family-level Beta-smoothed success estimate to estimate the probability of an informative group, divided it by measured generation-plus-verification cost, selected Train tasks without replacement, and logged probabilities and prior information. This is a small heuristic, not an optimality claim. Development and held-out tasks were ineligible for scheduler selection.

Each arm ran two seeds. Per seed the budget allowed six initial groups of four completions and three second-stage groups of four, for 36 completions per seed. Each scheduler therefore used 72 completions, 18 groups, and 360 attempted verifier cases. The same reward and grader were used throughout.

| Arm | Passes | Informative groups | Completion tokens | Generation (s) | Measured verifier time (s) | Unmeasured cases | Unique sources | Groups / 1k tokens | Groups / active minute* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 18/72 (25.0%) | 3/18 | 15,924 | 415.285 | 50.216 | 0/360 | 67 | 0.1884 | 0.3867 |
| Informative-per-cost | 22/72 (30.6%) | 3/18 | 16,643 | 431.911 | 51.523 | 5/360 | 68 | 0.1803 | 0.3723 |

\*Active minute is measured generation plus measured verifier case wall time. It excludes any queueing or unrelated host time.

Both arms found one informative group at seed 20261116 and two at seed 20261117, with the same informative task groups and the same matched parameter checkpoints across the two schedulers. The adaptive arm's higher raw pass count is not a causal comparison: the scheduler sampled a different task mix. It did not produce more informative groups under the matched completion budget and was less efficient on both reported cost denominators. The descriptive Wilson interval for 3/18 is wide (about 0.058–0.392) and does not account for task-level clustering.

## 4. Real Qwen training and Development behavior

The run used the pinned Qwen3-4B 4-bit MLX/Metal model, temperature 0.7, top-p 1, top-k 0, group size four, and zero policy lag. Sampling used the exact untruncated temperature-softmax distribution recorded with the completion log probabilities. Rewards remained binary and parent-verified. Zero-variance groups were skipped.

Each scheduler arm applied three real optimizer updates across its two seeds: update 1 in both seeds, plus update 2 for seed 20261117. Seed 20261116's second batch had no informative groups and correctly produced no update. Per arm, the effective trained completion-token count was 1,995 (720 for seed 20261116 and 826 + 449 for seed 20261117). For all applied updates, log-ratio means were 1.0, maximum absolute log ratio and approximate KL were 0, and clipping fraction was 0. Pre-clip gradient norms were 2.073 and 1.957 on update 1, and 2.244 on update 2. The resulting adapter digests were identical between schedulers for each matched seed:

- Seed 20261116: `37d250ffd7ef4c810efa6eb533cf089991d674fa2c85cf5ea998c5fb761b38b0`
- Seed 20261117: `2c0d547d909c4eeda069a1641ed7314e11cac26090a9aaceb2ef22e680fd21bd`

Adapter reload reproduced the original Development response hash and reward exactly. On the same four Development tasks, every pre-update and post-update policy remained at 2/4. The promotion gate rejected all four runs for insufficient evidence and no measured Development gain. This is real optimization, but a changed parameter digest is not evidence that the policy learned a useful capability.

The two arms' task mixes and completion pass counts differed, so pass-rate differences cannot be attributed to the scheduler. No independent held-out set was used, and no capability or generalization gain is established.

## 5. Durable recovery and trajectory provenance

The actual training path included model sampling, candidate isolation, trusted grading, exact sample persistence, trajectory-group provenance, behavior log probabilities, GRPO update, and checkpoint publication.

For informative-per-cost seed 20261116, the process received SIGKILL after publishing update 1's checkpoint but before its learner acknowledgment. Recovery verified all 24 sample identities and their trajectory consumption, reconstructed the missing acknowledgment, retained one effective update, and recorded no duplicate update. Recovery took 2.524 seconds. The recovered parameter digest was `10d5b1f7accc046551c127fec4920289e63352a417bd181c1da0984bb692876e`; the optimizer digest was `6439c27c87d561e0af0c01a2515f27e0b2febfeab7a716239e6ac37dda686ea6`.

Each run also persisted its exact second-stage samples, rewards, and behavior log probabilities before a locked SIGKILL boundary. Recovered state and replay-control state matched bytewise at the parameter and optimizer digest level. Where the batch had zero informative groups, the optimizer stayed at step 1 with unchanged hashes. Where it had one informative group, replay applied exactly one step and reached the same step-2 hashes. These records distinguish replay computation from duplicate effective commits.

The experiment did not inject SIGKILL during the optimizer kernel, stale-verifier behavior, or interruption between verification and promotion. Freshness was enforced with policy lag zero; a stale-verifier negative control remains future work. Adapter and optimizer weights are intentionally not in the PR; their local hashes are inventoried. Full long-horizon database event traces are provided as JSONL exports and can be checked with the evidence verifier.

## 6. Long-horizon transfer cohort

The prospective Development cohort contained four synthetic repository-repair tasks and a fixed 14-turn ceiling, with the same controller and task seeds for the unchanged base policy and the two-update seed-20261117 adapter. A post-hoc review found that the original primary metric counted a trusted hidden-grade pass even when the recorded public tests failed and the episode stopped at its budget. The controller's `finish` gate does not cover budget-stopped episodes. The metric and committed summaries now require both public-test success, when public tests are configured, and a passing trusted grade. This correction uses the public-test and grade outcomes already recorded in the event ledgers; no model inference or grading was rerun. The original locked runner is retained byte-for-byte in the run-source snapshot, and the correction is documented in [the public-gate correction record](../research/vare_ig_flagship/evidence/long_horizon/public-gate-correction.json).

| Task | Base final success | Trained adapter final success | Trusted grade only |
|---|---:|---:|---:|
| JSON pointer lookup | Fail | Fail | Fail / ungraded |
| Layered option lookup | Fail | Fail | Fail |
| Lexical course planner | Fail | Fail | Pass |
| Resource window coalescer | Fail | Fail | Ungraded |
| **Public + trusted task success** | **0/4** | **0/4** | **1/4** |

Both runs retained 81 hash-chained events and 56 committed agent steps. The base generated 6,588 output tokens, consumed 87,205 prompt tokens, made 41 tool calls, and took 405.417 seconds overall; generation time was 392.156 seconds. The trained arm generated 4,649 output tokens, consumed 63,901 prompt tokens, made 33 tool calls, and took 312.958 seconds overall; generation time was 302.316 seconds. Generation represented about 96.7% of each run's wall time. These are descriptive trajectory differences on four tasks, not a speedup claim.

Tool-call validity was 1.0 in both arms. Each arm had two episodes with no source change, so only half of the task episodes reached a trusted correctness grade. The fixed controller also assisted both arms: it ran public tests automatically and recovered bare edits; it issued two finish refusals for base and one for trained. Stop reasons included malformed edits in both arms, budget exhaustion, and one trained no-tool-call-after-edit stop. Neither policy resumed a failed agent action; measured agent recovery and resumed episodes were zero. The exact paired task outcomes did not change, so the paired success difference was 0/4 (McNemar p=1.0, descriptive only).

The traces include every stored state transition and are exported with their event hashes. Base database hash: `094374d656ba83eb4930ea61e3716630c6dfb14138a8b863922fb5a6426ff36e`; trained database hash: `80b88dd30663eac16fb4ed2b7aceb693bec47b9c3b8d1b07f2543816d7a4067a`.

## 7. Inference cost experiment

A paired same-prompt prefix-cache experiment ran eight cached and eight independent generations across two Exploration prompts. Output text and sampler probabilities matched exactly, and cache ownership checks passed. Mean task-group speedup was 1.0337x against a predeclared 1.02x threshold. This narrow result enabled cache reuse for the subsequent training workload. It is not a general MLX speedup estimate; it covers two prompts and eight pairs. The measured long-horizon runs show generation dominates this agent workload, but queue-inclusive p50/p95 latency, asynchronous verifier throughput, and an async-versus-sync comparison were not measured.

## 8. Implementation and repository integration

The branch adds:

- A 26-contract VARE-IG task pack, informative-group scheduler, verified group replay/provenance checks, and MLX shared-prefix reuse.
- Frozen exploration, cache, training-amendment, per-seed recovery, and long-horizon protocols.
- Scripts for task reference/mutant auditing, exploration regrading, protocol freezing, real MLX training, recovery reconciliation, update evaluation, and long-horizon cohort execution.
- A version 6 continuation inventory that adds the four new learning/cache source files while preserving the recorded v1–v5 inventories and hashes.
- A promotion-gate accounting fix for null or invalid verifier timings, plus a portable evidence verifier that checks file hashes, source snapshots, protocol copies, and both event ledgers.

The exact 14 source files in the training run locks are retained under [run_source_snapshot](../research/vare_ig_flagship/run_source_snapshot/). The four runs used commits `243cac18c0c8d9cd9065065a1ab52f732302c0a0` and `a4f7d44f2750098b648df545d914ca43136c0953`; those locked file bytes are identical across the two commits. The post-hoc public-gate correction changes current metric code but does not rewrite the run-source snapshot or historical protocol hashes.

## 9. Research contribution and relation to prior work

This is an engineering and empirical contribution, not a new GRPO or difficulty-sampling algorithm. [DeepSeekMath](https://arxiv.org/abs/2402.03300) introduced GRPO's group-relative baseline. [DAPO](https://arxiv.org/abs/2503.14476) already filters zero-variance prompt groups and adds dynamic sampling. [Online Difficulty Filtering for Reasoning-Oriented RL](https://aclanthology.org/2026.eacl-long.30/) studies intermediate-difficulty selection and reward sparsity directly. VARE-IG applies a family-level, verifier-cost-aware selection heuristic to a small synthetic coding distribution and integrates it with actual MLX rollouts and durable recovery.

The most useful result is a negative one: **having a scheduler that targets informative groups does not guarantee a better learning-signal yield or capability outcome.** Under this task pack and small budget, it tied uniform sampling on informative groups, produced no Development improvement, and both policies failed every long-horizon task under the public-plus-trusted success rule. The run also shows why task selection needs to be evaluated on matched cost and policy outcomes—not just the raw count of passes or whether an optimizer step happened.

## 10. Limits and next experiments

The main limitations are the small task slate, synthetic semantics, two seeds per arm, wide uncertainty, one truncated candidate, and no untouched held-out coding benchmark. The family-level pilot estimates are noisy and should not be treated as stable task difficulty labels. There was no group-size, temperature, or reward ablation. The adapter's short function-task training did not improve the four-task agent cohort. Recovery coverage is meaningful but does not include an optimizer-kernel kill or stale-verifier injection. The prefix reuse result is narrow, and asynchronous latency/throughput was not measured.

A next study should expand to independently audited real repository contracts, increase seed/task coverage, freeze a larger unseen evaluation pool before scheduler tuning, compare the current heuristic with a group-level dynamic-filtering baseline, and measure end-to-end queue latency. Long-horizon learning should train on multi-step tool trajectories if the target claim concerns agent recovery or repository-level editing.

## 11. Reproduction and verification

From a checkout with the pinned project environment and Docker/Colima available:

```bash
# Verify committed experiment files, locked source snapshot, and both trace hash chains.
python scripts/verify_vare_ig_evidence.py

# Re-run the reference/mutant audit into a fresh path.
DOCKER_HOST="${DOCKER_HOST:?set Docker/Colima socket}" \
  python scripts/audit_vare_ig_task_oracles.py \
  --output work/vare-ig-task-oracle-audit-reproduced.json

# Focused unit and integration coverage.
DOCKER_HOST="${DOCKER_HOST:?set Docker/Colima socket}" \
TMPDIR="${TMPDIR:-/tmp}" PYTHONPATH="$PWD/src" \
  python -m pytest -q \
  tests/test_vare_ig_curriculum.py \
  tests/test_vare_ig_verified_rollout_replay.py \
  tests/test_vare_ig_long_horizon_cohort.py \
  tests/test_vare_ig_evidence_manifest.py \
  tests/test_long_horizon_eval.py \
  tests/test_e2e_rl_continuation.py

mypy src/vare
```

The CI lint command is the `ruff check` invocation in [.github/workflows/ci.yml](../.github/workflows/ci.yml); all VARE-IG scripts and modules are also linted during local validation. The model runs use the individual per-arm/seed lock files in `protocols/vare_ig_qwen3_mlx_dev_v3b/`; their recorded SHA-256 values are in each evidence run's `run_meta.json` and in `summary.json`. The run uses the portable copy of the task protocol only after its path placeholders are resolved and its hash is checked against the redaction manifest. Do not edit an existing lock to start a new experiment: freeze a new protocol and new run IDs for new work.

For long-horizon reproduction, use `scripts/vare_ig_long_horizon_cohort.py run --arm base --protocol protocols/vare_ig_long_horizon_dev_v1.json --output <new-output-root>`. The trained command also requires `--adapter <adapter-export>` and `--training-run <run-directory>`. The adapter weights are local-only; their SHA-256 values are recorded above and in `summary.json`.

## 12. Integration status

The starting main SHA and research evidence are frozen above. Required PR checks and the merged main SHA are recorded in the companion final report artifact after exact-head CI completes. The committed repository report does not guess a merge SHA in advance.
