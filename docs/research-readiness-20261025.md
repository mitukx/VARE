# Coding research readiness and integrity integration — 2026-10-25

**Decision: NO-GO for new model training, held-out evaluation, real-coding async comparison, or a learned RSI selector.** This cycle audits retained task/exposure records and integrates a missing recovery-auditor correction. It measures no model-quality improvement.

The repository snapshot has main `f3ab90a36e8724441b6b9555255f8f75a0a4d529` and 23 open PRs. `main` requires one approving review plus the required CI contexts; automated review alone does not satisfy a missing human-review requirement. No PR was merged.

**Task evidence.** The original 15-task source pool has 3 Train, 3 Development, and 9 held-out definitions. All 30 retained baseline/fix log hashes match their frozen catalog. However, Transformers #49412 is present in nine subsequent Qwen3-4B Development inference receipts on the exact same `cb33194ad6152bd9fad6305378d92db385dd7b32` baseline. The alias is `transformers-49412-fusion-registration-idempotence-v2`; seeds are 201–203, 401–403, and 601–603. This is positive evidence of task exposure, not evidence that hidden labels were trained on. The source catalog is preserved; the task is excluded prospectively from fresh held-out use.

**Gate.** At most 8 originally reserved families remain before further clearance; 15 eligible held-out tasks are required. This audit admits zero: the remaining entries are source-level reproductions, with no complete current isolated-grader/environment/exposure certificates established here. A path search across 123 locally retained origin tips found the #49412 receipts; absence of a match is not proof of no external or historical exposure. Later #122 and #125 tasks are consumed Development tasks and add zero held-out tasks. SWE-Bench Pro V2 remains NOT_ELIGIBLE without the exact contaminating candidate and paired grades.

| Source task | Allocation | Current blocking reason |
| --- | --- | --- |
| `transformers-45852-4d-mask` | development | Frozen Train/Development allocation |
| `transformers-48319-stale-best-checkpoint` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `transformers-49208-resume-metrics` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `transformers-49412-fusion-idempotence` | heldout | Consumed by nine retained Development model attempts; retired from fresh held-out use |
| `mlx-4662-padding-input-gradient` | train | Frozen Train/Development allocation |
| `mlxlm-1795-moe-routing-gradient` | train | Frozen Train/Development allocation |
| `mlxlm-1902-continuation-score-truncation` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `mlxlm-1904-tool-arguments` | train | Frozen Train/Development allocation |
| `mlxlm-1912-top-p-min-token` | development | Frozen Train/Development allocation |
| `mlxlm-1921-padded-logprobs` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `mlxlm-1926-falcon-cache-gradient` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `mlxlm-1935-zero-max-tokens` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `mlxlm-1938-phi3small-logits-mask` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |
| `mlxlm-1971-mimo-value-scale` | development | Frozen Train/Development allocation |
| `mlxlm-1979-ssm-padding-mask` | heldout | Source-only qualification; current isolation/environment/exposure certificate incomplete |

**Selected executable correction.** At the starting heads, #127 (`ecc5570a…`) and #128 (`acca0700…`) lacked #126 commit `c728b9f2ece5dbd1f08878520b0efe5e17ca3b48`. The missing code enforces the frozen behavior-logprob limit and binds recovered SQLite consumption/acknowledgments and the durable active pointer to the checkpoint. The explicit parent integration commit `25895688b8d605eca98f1fca4ec9dceaa1db7f9d` brings those checks to #127; the change is propagated to #128. Frozen v7 protocols, consumed rollouts, raw results, and the v1 crash auditor are unchanged. This is propagation of an existing correction, not a new reliability or learning result.

Eight relevant regression files complete with **40 passed, 1 skipped**, including stale pre-ack snapshots, wrong acknowledgment heads, stale durable pointers, non-finite/alignment-limit failures, and output ownership. Tests use constructed corruption controls; no model inference or fresh optimizer update was performed. Exact final-head CI and independent review remain required after branch propagation.

**PR dependency map at the initial snapshot.**

`main → #99 → #100 → #101 → #102 → #103 → #104 → #105 → #106 → #107 → #108`  
`#99 → #114 → #117; #101 → #118; main → #120, #121, #88`  
`main → #122 → #123 → #124 → #125 → #126 → #127 → #128`

The parent heads were not ancestors at #99→#100, #102→#103, #106→#107, #107→#108, #99→#114, and #126→#127. Only the final edge is addressed here. The older gaps include omitted grader/controller corrections and require separate review; the entire stacks must not be merged mechanically. #99 and #102 still have unresolved P1 threads; #107 has an unresolved protocol-source P1. #117 has unresolved outdated review threads that require adjudication, not automatic closure.

| PR | Starting head | Draft | Review/CI evidence |
| --- | --- | --- | --- |
| [#88](https://github.com/mitukx/VARE/pull/88) | `ae6e469769127d042623f31dd8cc8247b1babba6` | True | 20 successful checks; 0 failed; 3 unresolved threads |
| [#99](https://github.com/mitukx/VARE/pull/99) | `762a5a65598756014d957a206ba1548cb19a348f` | True | 20 successful checks; 0 failed; 2 unresolved threads |
| [#100](https://github.com/mitukx/VARE/pull/100) | `4c03f12992c6ff45a6e5342a58e6b1927a8d28d2` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#101](https://github.com/mitukx/VARE/pull/101) | `aef27e3b7bb3dcadb4bd8d8ee1ca6bb6ebb7634d` | True | 29 successful checks; 0 failed; 0 unresolved threads |
| [#102](https://github.com/mitukx/VARE/pull/102) | `0ba50de60f2013ea1838bb44e1068c44c26fd6c4` | True | 20 successful checks; 0 failed; 3 unresolved threads |
| [#103](https://github.com/mitukx/VARE/pull/103) | `031255fe3f2bf5b2913a393e2da3fb26d2a01d13` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#104](https://github.com/mitukx/VARE/pull/104) | `02684dcce4c0a5474256b1d97a2adbda879388d2` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#105](https://github.com/mitukx/VARE/pull/105) | `a6c3071b5f47e988fe6c3330922c45833be04694` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#106](https://github.com/mitukx/VARE/pull/106) | `075095c650ac372ff8d686e4461314feda26444a` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#107](https://github.com/mitukx/VARE/pull/107) | `4bdca9b51640e60c47c3aefe63c1c8f12834de85` | True | 19 successful checks; 0 failed; 1 unresolved threads |
| [#108](https://github.com/mitukx/VARE/pull/108) | `710dbc0d28aa33a78521a29f067c9a9493f6c56d` | True | 19 successful checks; 0 failed; 0 unresolved threads |
| [#114](https://github.com/mitukx/VARE/pull/114) | `8257a4dfe2106055a048ddd4b1520a08c7e4d658` | True | 19 successful checks; 0 failed; 0 unresolved threads |
| [#117](https://github.com/mitukx/VARE/pull/117) | `6709aed996ce5ec315cf7a87930350dca6c0ee20` | True | 19 successful checks; 0 failed; 4 unresolved threads |
| [#118](https://github.com/mitukx/VARE/pull/118) | `552d6f61187926d78719aad8c4c7ec9e8663c242` | True | 20 successful checks; 0 failed; 0 unresolved threads |
| [#120](https://github.com/mitukx/VARE/pull/120) | `58ecd9c9727a1544a5a08c9c250f1f36dd9a5f8d` | True | 20 successful checks; 0 failed; 0 unresolved threads |
| [#121](https://github.com/mitukx/VARE/pull/121) | `a937f4ad6eacb63d679952f7a05d39d04afab59a` | True | 19 successful checks; 0 failed; 0 unresolved threads |
| [#122](https://github.com/mitukx/VARE/pull/122) | `37b31519818b5dfbf245075cc541d6c8a38b4099` | False | 28 successful checks; 0 failed; 0 unresolved threads |
| [#123](https://github.com/mitukx/VARE/pull/123) | `a02d6cb64f2531cbe6152266b431d8a466f0bdbc` | False | 19 successful checks; 0 failed; 0 unresolved threads |
| [#124](https://github.com/mitukx/VARE/pull/124) | `af4bc1f862f2b8b2d0564cd8731b81129c1195ec` | False | 19 successful checks; 0 failed; 0 unresolved threads |
| [#125](https://github.com/mitukx/VARE/pull/125) | `a476feeed7ee5d711b3741b0a0ffa89f7c29cd76` | False | 28 successful checks; 1 failed; 0 unresolved threads |
| [#126](https://github.com/mitukx/VARE/pull/126) | `c728b9f2ece5dbd1f08878520b0efe5e17ca3b48` | False | 20 successful checks; 0 failed; 0 unresolved threads |
| [#127](https://github.com/mitukx/VARE/pull/127) | `ecc5570a4e2275ecb78d8367f2bd383fbd197cac` | True | 19 successful checks; 0 failed; 0 unresolved threads |
| [#128](https://github.com/mitukx/VARE/pull/128) | `acca07004eb9a44cf23d31ff0d68929b7ea3fc1d` | True | 19 successful checks; 0 failed; 0 unresolved threads |

#125 retains an earlier failed Python 3.13 check at the same head: a 0.1-second injected-worker lease expired before its first step. Later push and PR CI and E2E runs at that head succeeded. This is retained timing-sensitive fixture evidence, not a newly discovered model-state recovery failure.

**Ranked research gaps.** Eligible disjoint task certificates come first; independently verified positive repair trajectories come second; checkpoint-bound held-out outcomes after an update come third; a coding-relevant verifier transition and matched async study follow those gates. The prospective #127 evaluation journal does not reconstruct its missing historical evaluation outputs. The v7 4/8, 1/8, 0/8 sequence and the failed reward-shift gate remain NO-GO. The 504-round study and retained optimizer replays are not rerun as new research.

**Costs and uncertainty.** The focused retained-record reproducer took 1.23 seconds wall, 0.37 seconds user CPU and 0.75 seconds system CPU, with 18,169,856 bytes maximum RSS and zero process swap events reported by `/usr/bin/time -l`. The wider 123-tip discovery scan took 105.11 seconds wall / 960.92 seconds total CPU and reported 1,695,825,920 bytes maximum RSS. These are audit costs, not training costs. Counts describe a finite inventory, so a sampling interval is inapplicable; unknown exposure and incomplete certification are explicit exclusions. Paid spend, model downloads, fresh inference and training cost are zero.

**Reproduce the task decision.** From a full VARE checkout with the two recorded historical commits available:

```sh
python scripts/audit_coding_task_exposure_20261025.py --output /tmp/new-task-readiness.json
```

The auditor uses Git objects with replacement refs disabled, verifies frozen input and all 30 log hashes, validates the nine model receipts, and refuses an existing output file. See [`task-readiness.json`](../results/research-readiness-20261025/task-readiness.json), [`pr-assessment.json`](../results/research-readiness-20261025/pr-assessment.json), the resource log, and regression XML in the same directory.

**Gate decisions.** Data: NO-GO. Grader certification for a fresh held-out pool: NO-GO. Learner and real-coding async comparison: NOT_ADMITTED. Existing narrow auditor integration: local regressions pass, subject to final CI/review. RSI selection comparison: NOT_ADMITTED.

**Highest-value next action.** Qualify at least seven additional disjoint held-out task families and complete the eight surviving source-task certificates before freezing a training comparison; training/development positives must be obtained separately. Do not use the reserved eight to tune rewards or task selection.
