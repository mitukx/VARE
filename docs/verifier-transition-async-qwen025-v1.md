# Verifier-transition asynchronous GRPO: Qwen2.5-0.5B, three-seed systems experiment

Date: 2026-10-29 JST. This is a controlled synthetic systems result, not coding capability or held-out model improvement.

## Question and scope

Does using nonconstant reward from a superseded verifier change an actual GRPO update, and can use-time rejection, trusted regrading, and process recovery control that change? The cheapest preflight checked the reward complement, durable hash chain, replay and GRPO contracts (29 tests passed). The protocol and executable/auditor hashes were committed at `fcfcb248` before inference. The frozen protocol SHA-256 is `20d8c6b8d3292ae1379f403ddb700a45d76af91a96d90f39d561937e26e6452c`.

The actor is Qwen/Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, model weight SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. CPU fp32, Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, one thread per actor/learner. RVL revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72` supplies the actual clipped token-level GRPO and AdamW implementation. VARE supplies whole-group replay, use-time lag admission and group-relative advantages. Initial VARE main was `5acc077969d60141827a259495ca77792dbb7f44`.

The pretrained model is frozen. Its vocabulary-logit bias is trainable; a consistent A/B/C/D mask is used by sampling and training likelihoods. Only four coordinates receive nonzero gradients. This is neither LoRA nor full-model training. Each rollout is one categorical token, drawn from actual Qwen logits at temperature 2. The actor deliberately remains at initial policy version 0; all arms permit policy lag up to 5 to isolate verifier freshness. This does not test asynchronous policy deployment or long coding trajectories.

Verifier v0 rewards A/B and v1 rewards C/D. This is a declared synthetic objective reversal, not a claim about repairing a real oracle. Gradients are expected to oppose under this intervention; the experiment tests the model/optimizer/provenance execution of that counterexample. Independent evaluation uses three separate synthetic prompts and conditional probability P(C)+P(D) at temperature 1. It is not an independent coding task metric.

## Frozen comparison

Seeds 101/202/303; five groups per seed/arm; 16 samples per group; learning rate .01; AdamW defaults; GRPO clip .2; population reward normalization epsilon 1e-6; gradient norm cap 1. All groups had nonconstant rewards; no replacement group was needed. Group 0 updates under v0, then the current verifier becomes v1 before delayed groups 1/2 arrive; groups 3/4 carry v1 rewards.

| Arm | Treatment |
| --- | --- |
| A | Demand-triggered synchronous actor, current verifier |
| B | Async actor; uses old v0 rewards for groups 1/2 after transition |
| C | Async actor; VARE rejects those complete stale groups, retains artifacts/receipts |
| D | Async actor; trusted v1 rescores/re-certifies stale groups, then VARE admission |
| E | D plus four actual SIGKILL cutpoints per seed |

Every arm uses the same seeded draws, initial actor/learner, prompts, input-group budget and resource caps. A/D/E therefore have matched effective update budgets; C's lower update count is the cost of its rejection treatment. Fixed A/B/C/D/E execution order confounds timing with host/cache effects. Rates are descriptive load-inclusive measurements, not universal asynchronous speedup claims. The 1,200 emitted records contain 240 unique seeded draws replayed across the five matched arms; only three seeds are independent experimental units.

## Completed results

| Arm | Effective commits | Stale-verifier commits | Mean v1 evaluation | Trajectories/s | Admitted trajectories/s | Queue p50 / p95 seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 15 | 0 | .098459 | 4.378 | 4.378 | 2.584 / 3.005 |
| B | 15 | 6 | .092391 | 4.527 | 4.527 | 5.568 / 9.622 |
| C | 9 | 0 | .096926 | 5.690 | 3.414 | 2.592 / 5.215 |
| D | 15 | 0 | .098459 | 4.569 | 4.569 | 5.408 / 9.124 |
| E | 15 | 0 | .098459 | 2.366 | 2.366 | 11.429 / 24.490 |

Queue quantiles use the 15 group-level observations per arm, from generation start through commit/rejection. They include recovery and are not 240 independent latency samples. All 69 effective commits changed an actual trainable adapter. There were 72 actual optimizer computations: three uncommitted computations were repeated after interruption, with no duplicate effective commit.

| Seed | B–A final adapter L2 | B–A v1 evaluation difference | First stale-gradient cosine, B vs A |
| --- | ---: | ---: | ---: |
| 101 | .085207 | -.005782 | -1.000 |
| 202 | .072289 | -.005819 | -1.000 |
| 303 | .084241 | -.006602 | -1.000 |

The mean evaluation difference is -.006068, range [-.006602, -.005782]. These are descriptive three-seed synthetic results, without a broad learning/significance claim. Regrading recovers the same parameters, Adam state and CPU RNG as A in every seed; E exactly matches uninterrupted D and A. C removes all six contaminated updates but admits only 144 of 240 input trajectories, explaining the utilization tradeoff.

## Faults, audit and resources

Each E seed kills the worker after durable group-1 write before acknowledgment; the fresh worker reuses the exact artifact. Learner faults occur after group-2 optimizer application/precommit, group-3 checkpoint/pre-publication, and group-4 publication. Three worker and nine learner SIGKILL exits were observed by the parent, followed by distinct fresh process IDs. Recovery restores adapter, actual Adam tensors/counters and RNG; committed groups are not trained twice.

The independent auditor imports neither runner nor trainer. The historical v1 receipt recorded **897 PASS**, and archive extraction reproduced that receipt byte-for-byte. External review then exposed three missing bindings: an earlier valid active pointer could pass, gradient differentiation trusted the event's self-reported starting bias, and the reported evaluation scalar was not recomputed. The v1 receipt/source are preserved as incomplete historical audit evidence.

Appended auditor v2 anchors every gradient attempt (including uncommitted repeats) to independently replayed pre-update parameters, binds active path/hash/policy version to the final accepted commit and actual checkpoint bytes, and independently reconstructs evaluation prompts/probabilities/means from the frozen model plus validated adapter. **1,236 checks passed** on the archived real run. All reported primary numbers remain unchanged. Three adversarial copies (old pointer, false gradient origin with a valid rehashed journal, and corrupted evaluation mean) each passed v1 and each failed v2 at its intended semantic check. Their mutations, full PASS/failure logs and reproduction script are retained. No model samples or optimizer updates were rerun for this correction.

The original and v2 auditors ran in 6.31 and 11.21 seconds, respectively, each peaking at about 3.38 GB RSS. Gradient replay uses a 2e-5 fp32 absolute tolerance; gradient-origin and Adam replay use 1e-7, while A/D/E checkpoint-state comparisons are exact. V2 is a post-observation audit strengthening, not a rewritten preregistration or a new verifier rescore.

Maximum sampled combined child RSS was 6.37 GiB; swap growth was zero in every case. Inference was CPU only; Metal allocation was zero. Total case wall time was 305.17 seconds. Caps remained 16 GiB process RSS, 512 MiB swap growth, 4 GiB initial/2 GiB hard available RAM, 10 GiB disk and 1,200 seconds. Mean compressed adapter/optimizer/RNG checkpoint I/O was approximately .015 seconds; largest checkpoint was 7,544 bytes. These sparse-bias checkpoint costs do not represent full-model checkpoint costs.

Each case used 80 original reward decisions; D/E added 32 trusted regrades. All cases also computed 80 current-verifier shadow labels for diagnostic receipts. Per-call verifier wall cost was not separately instrumented (**NOT_MEASURED**). Recovery timing observations in `metrics-v1.json` bind post-run filesystem fault-marker mtimes to durable terminal timestamps; they are weaker than native monotonic cutpoint timers and are not a precise recovery microbenchmark.

A separately frozen publication-control protocol used new synthetic prompts after the primary comparison. It promoted final D over its group-0 incumbent in 3/3 seeds. A declared -20 C/D bias negative control was rejected in 3/3, preserving the active pointer. This post-primary control validates publication/rollback plumbing; it does not independently confirm learning or add optimizer updates.

## Repository and eligibility audit

Live reconnaissance found #143 already merged by mitukx at `41512fb1` without approval reviews, followed by #140 at `5acc0779`. Those are pre-sprint owner actions. Current protection requires zero approving reviews, enforces admins and nine strict CI contexts; no rulesets were present. Prior assumptions about a mandatory one-review gate are not current settings. An explicit human-review requirement remains in the #142 split-collision discussion and was not treated as satisfied.

The starting inventory contained 19 open PRs / 9 Drafts, including the independently created #146 v10 measurement PR. A machine-readable ledger records exact heads, bases, CI and implementation files. #139 remains a large stacked branch; this experiment is extracted onto a new main-based branch instead of merging its history.

#145 received prospective protocol v2 at `1226dedc`: public-test availability, allowlist and container requirement are family-specific. TRL #7913 uses syntax feedback followed by independent grading; Mask2Former #49703 is blocked because public-test metadata is unpinned. Frozen v1 is unchanged. Eight v1/v2 tests passed and the review thread received evidence. The protocol still does not authorize model inference; no diagnostic coding campaign was run.

#144's inventory artifact and test were independently reproduced. The prospective source/grader-certified count remains **8/15**, fully model-eligible count remains UNKNOWN, and pretraining exposure remains UNKNOWN. #1880 and #1926 remain excluded. Readiness digest `ead67871…` was rechecked; its conflicting citation and the fix/split/theme discrepancies remain OPEN_UNVERIFIED. No held-out source, hidden grader or known-fix diff was opened.

The first #144 CI head failed a pre-existing timing test because an insufficient workload correctly returned no adjusted ratio, while the test unconditionally required one before its own rejection branch. `b2d56b12` makes adjusted-statistic assertions conditional on sufficient workload. Production timing, thresholds, raw timing checks and unconditional rejection are unchanged. Four local Colima tests passed, including the real import-precompute case and two deterministic gate cases. The original failed CI log is retained.

After the experiment, a separate launcher regression found that the failure recorder could replace failure.json when invoked on an existing partial output. The current launcher rejects an existing output before entering that recorder; a subprocess regression proves the old failure bytes remain unchanged. No completed experiment used a pre-existing output. Frozen experiment source remains at fcfcb248; the current launcher and regression now differ from that frozen source, so they cannot silently rerun the original protocol hash. The final relevant suite has 30 passing tests.

Sprint integration moved main from `5acc077969d60141827a259495ca77792dbb7f44` to `50fd8c57788df6ef9aa90bcf483282721e696ae9` by merging #144 after 19/19 exact-head checks and no unresolved review threads. The 8/15 correction and fail-closed timing-test fix are now on main. #145 was updated onto that main; its review confirmation remains pending. #142 received a split-audit status reply and stays Draft/unmerged. No PR was closed without merge. Before creating this separate research-record PR, open/Draft counts were 18/9 (start 19/9). This research PR adds one Draft; final live CI/counts are recorded in its description. No green historical head is substituted for a later head's result.

## Reproduction and evidence

- Frozen protocol/source hashes: `results/sprint_xii_20261029/protocol-v1.json` and sidecar; source freeze commit `fcfcb248`.
- Lossless raw data: three per-seed tar.gz archives; every regular member was roundtrip byte-compared. `raw-archive-manifest-v1.json` binds member identities. Local `run-001/` remains outside Git; archives retain all raw trajectories, process logs, journals, receipts and compressed torch states.
- Extract archives into a **new temporary directory** using Python `tarfile.extractall(filter='data')`. Copy the retained top-level `run-001-summary.json` as `summary.json`. Run `audit_async_verifier_v2.py --receipt <new-receipt.json>` on that temporary tree with the pinned model/dependencies, so historical audit files are not overwritten. V2 refuses to overwrite an existing receipt and leaves input evidence unchanged. The v1 auditor is retained only to reproduce its historical behavior and the counterexamples. `summarize.py <temporary tree>` recomputes the table without inference.
- Model cache and VARE/RVL roots in the frozen protocol are local paths. Relocated audit requires an explicitly recorded audit-only protocol copy with the same source/model hashes and adjusted paths. Do not rewrite the frozen protocol. A new model-generation campaign requires a separate prospective protocol and exposure accounting.
- Historical primary command (use the source snapshot at `fcfcb248`, not the subsequently corrected launcher): `/tmp/vare-sprint-xi-cpu/bin/python scripts/sprint_xii/async_verifier.py run --protocol results/sprint_xii_20261029/protocol-v1.json --output <new-absent-output>` with exact pinned sources. Do not reuse the completed output path or present repeated draws as new confirmation.

## Decisions

| Item | Status and evidence boundary |
| --- | --- |
| Nonconstant stale-reward optimizer comparison | RUN / GO for this synthetic systems question |
| Trusted regrading and actual crash recovery | RUN; exact matched checkpoint state in all three seeds |
| Publication/rollback control | RUN; 3/3 positive and 3/3 negative controls |
| Long coding-agent trajectories / actor policy refresh | NOT_RUN |
| Real external coding repairs in this sprint | 0; NOT_RUN |
| Coding SFT/GRPO / blind held-out improvement | NO-GO; 15-family gate unmet |
| Adaptive verifier-refresh research loop | NOT_RUN; fixed schedule only |
| General LoRA/full-model learning or production throughput | Not established |

The next useful experiment is a less adversarial verifier drift workload with interleaved arm order, refreshed actor checkpoints, equal accepted-update and wall-budget analyses, native cutpoint timing and verifier-call timing. This counterexample shows that stale verifier semantics can change real optimizer state and that regrading/recovery can preserve a declared reference. It does not establish an optimal rejection/regrading policy or downstream coding improvement.
