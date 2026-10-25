# V7 verifier-drift research reassessment

## Decision

**NO-GO for another real-model learning or asynchronous comparison under the
current evidence.** The frozen v7.3 stop and the calibration's failed 0.15
mean-shift gate remain binding. The new analysis verifies that verifier changes
can alter the exact GRPO objective and AdamW candidate update on the saved
groups, but it does not establish improved or degraded task quality.

The next major obstacle is **B: a coding-relevant, independently meaningful
verifier transition**. The measured transition is a synthetic surface-scoring
example. It demonstrates mechanism sensitivity, but it does not justify a
coding-RL experiment. The old calibration also remains a specific NO-GO under
its unchanged gate (A); that gate was not tuned or crossed post hoc.

## Corrected evaluation evidence path

The P1 retention journal remains prospective. Each completed evaluation is
written immediately with its exact input/output, scores, pass decisions,
model/verifier identities, invocation ID, source and checkpoint hashes, and
timestamps. Failure and partial-stop records are retained. Writes are atomic,
non-overwriting, and hash-checked; fault tests cover interruption after an
evaluation, during promotion, and before arm completion.

The journal module now lives at
`scripts/verifier_drift_evaluation_evidence.py`. Keeping it outside
`src/vare/` restores the historical continuation auditor's exact 77-file
runtime inventory. The v7.3 runner, protocol hashes, and NO-GO evidence remain
unchanged. A local `python3.12 -S` import/help check and the focused journal,
schedule `e2e-system-cpu` because the net PR diff adds the prospective module
under `scripts/` and has no `src/vare/` change; the full workflow must be
dispatched on the final PR head and reviewed before the PR can leave draft.

The exact-head E2E run on `75a86532` passed, including continuation, SIGKILL
recovery, and pinned-RVL regression. Its independent review found that arm
wall time excluded the final evaluation and that two report writers could
overwrite retained inputs. The prospective runner now checks its wall budget
after final evaluation. The full scenario auditor now rejects output paths
inside audited inputs and writes with exclusive creation. The crash auditor's
original v1 source is hash-pinned by the frozen protocol, so a prospective v2
CLI adds those output protections while calling the unchanged v1 audit logic.
These fixes require another exact-head E2E run and review before the draft can
leave draft; no new learning run is authorized.

## Frozen calibration: independently reconstructed

All 32 saved outputs were checked against the frozen reward definitions and
input hashes. There are 11/32 strictly changed rewards (34.4%); mean absolute
disagreement is 0.1140625; 9/32 outputs move by at least 0.25. The original
0.15 mean-shift gate therefore still fails. Reward counts are identical under
both versions: Earth 7 positive/0 negative/1 zero; Elevated 6/0/2; Built 2/0/6;
Frozen 2/0/6.

Across the 112 within-group pairs, there are nine strict preference reversals,
all in Elevated (9/28 there; 0/28 in each other group). The source-bound RVL
group normalization changes centered advantages in every group, though the
Built change is only numerical-scale (mean absolute change below 0.000001).
No member changes advantage sign.

At the unchanged TinyStories-1M checkpoint, I recomputed gradients with the
pinned token-level clipped GRPO objective from the exact saved token IDs and
behavior log-probabilities. Gradient cosine similarity for v0 versus v1 is
0.942 (Earth), 0.744 (Elevated), approximately 1.000 (Built), and 0.971
(Frozen). Maximum behavior-logprob ratio error is below 0.000017. The model
weights are byte-identical before and after; **zero optimizer steps** were
applied. This measures the optimizer-input gradient, not a learning outcome.

The result is recorded at
`results/verifier_consistent_async_grpo_v7/reward-calibration-v1/postrun-analysis-v1.json`
(SHA-256 `de4a33a71337fbc9096d9f56baad9e315a3441ac34210f36e14b9bb766f24a75`).
Its metrics reproduced exactly on a second invocation; it binds the raw input,
protocol, auditor, analysis script, model manifest, and RVL objective hashes.

## Fixed authenticated replay with real optimizer states

The separate replay at
`results/verifier_consistent_async_grpo_v7/optimizer-replay-v1/` uses the same
32 fixed outputs, authenticated trajectory receipts, independently
recomputed rewards, and the pinned TinyStories/RVL checkpoints. It compares a
version-unaware v0 admission with a v1-aware admission. The aware path rejects
all four stale v0 groups at both group admission and the trainer boundary,
without an optimizer call, then admits all four groups re-scored under v1.

Each arm runs one real GRPO/AdamW candidate step per group from the same reset
checkpoint: four v0 and four v1 steps, all with Adam step count 1. The
candidate model and optimizer states are hashed; no candidate is promoted. The
v0/v1 AdamW parameter-update cosine is 0.775 (Earth), 0.604 (Elevated),
approximately 1.000 (Built), and 0.849 (Frozen). This shows that the tested
reward transition changes the actual one-step update direction in three of
four fixed groups. It is a scripted deterministic systems control on historical
synthetic data, not fresh on-policy learning, a model-discovered exploit, an
asynchronous comparison, or a capability result. The SQLite trajectory audit
passes with 64 completions, 64 steps, and zero effective duplicates. The run
inventory SHA-256 is
`43f22cf32ca76cea2655cf0d2f758a0679368c277df034542b5bcaa234701059`.
A second clean invocation reproduced all four candidate model and
optimizer hashes, train-step metrics, parameter deltas, and update cosines
exactly. Its inventory SHA-256 is
`0033db0893459c4ba9f6675fbb4e5505affd75a22599b4a2557b17d63e2c292b`.

Two earlier local replay attempts are preserved separately with explicit
failure records. One completed the optimizer branches but failed while
finalizing an unsupported store-close call; the other recorded a cosine over
absolute weights instead of update deltas. Neither output is used for the
reported comparison.

## Dependency-chain evidence and scope

PRs #122–#127 remain an open stack: #123 is based on #122, #124 on #123, #125
on #124, #126 on #125, and draft #127 on #126. None is merged. PR #122's image
artifact has an independently downloadable release and binds the contaminated,
source-only, and sanitized images. PRs #123 and #124 contain the bounded
counterfactual and scripted bandit evidence; the latter is not language-model
training. PR #125's separate streamed-GRPO artifact is published and its
updates were replay-audited, while its fresh coding-agent attempts were
negative. PR #126's complete rollout/checkpoint/SQLite crash run remains in the
local results tree. PR #127's 82 MB v7.3 run is also local-only; it stopped
before the verifier transition or asynchronous arm, and its held-out raw
evaluation records are missing. The prospective journal repair does not fill
those historical records.

This scope is narrower than existing benchmark-integrity work. SWE-Bench+
reported issue/comment solution leakage and weak tests; SWE-Bench Pro Verified
adds anti-hacking controls for gold-solution/hidden-evaluation leakage and
refines flawed tasks. Recent work on unearned passes audits trajectories and
replays exploit channels, with Git history a prominent example. Those papers
already establish the general integrity problem and hardening loop. VARE's
candidate incremental contribution is dependency source/bytecode provenance
plus counterfactual delegation controls; the current v7 replay does not test
that contribution or establish its novelty. Formal reward-hacking work also
distinguishes optimizing a proxy from improving the true objective. AReaL and
other asynchronous RL systems already study staleness and throughput; v7 never
reached a matched async comparison, so this result makes no asynchronous
systems novelty claim.

## Reproduction

From the VARE checkout, using the already pinned local model and RVL checkout:

```sh
python3.12 -S scripts/audit_verifier_reward_calibration.py \
  results/verifier_consistent_async_grpo_v7/reward-calibration-v1/pilot-1
python3.12 scripts/analyze_reward_calibration_directions.py \
  --output results/verifier_consistent_async_grpo_v7/reward-calibration-v1/postrun-analysis-reproduction.json
python3.12 scripts/run_verifier_consistency_optimizer_replay.py \
  --output results/verifier_consistent_async_grpo_v7/optimizer-replay-reproduction-1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3.12 -m pytest -q \
  tests/test_evaluation_evidence.py tests/test_verifier_drift_summary.py \
  tests/test_authenticated_verifier_drift_regression.py
```

The analysis/replay scripts refuse to overwrite existing output paths. Use a
new path for each reproduction. The final exact-head `e2e-system-cpu` workflow
and independent PR review remain release gates; PR #127 stays draft and
unmerged.

## Related work

- [Maintaining Benchmarks Against Increasingly Capable Agents: Detection and Remediation of Unearned Passes](https://arxiv.org/abs/2609.34262) audits pass trajectories, including Git-history solution access, and validates repairs through exploit replay and fresh evaluation.
- [SWE-Bench Pro Verified](https://arxiv.org/abs/2609.08149) studies gold-solution/hidden-evaluation leakage and task-quality defects, then hardens and refines tasks.
- [SWE-Bench+: Enhanced Coding Benchmark for LLMs](https://arxiv.org/abs/2410.06992) analyzes answer leakage from issues/comments and weak test acceptance.
- [Defining and Characterizing Reward Gaming](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html) formalizes the gap between proxy reward and true reward.
- [AReaL](https://arxiv.org/abs/2505.24298) studies fully asynchronous LLM reinforcement learning, stale rollouts, and throughput; this VARE result did not reach a matched async comparison.
