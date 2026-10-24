# V6 verifier-consistent GRPO and process-recovery results

## Result and decision

The preregistered two-seed synthetic verifier-drift pilot completed, followed by five actual learner-process `SIGKILL` injections. The matched VARE path rejected both stale groups and committed only fresh, verifier-consistent updates. The naive path committed one invalid mixed-label update in each seed. Every crash case recovered to the exact synchronous model, Adam, CPU RNG, and module-mode state, with one effective commit and no duplicate trajectory consumption.

This is a **systems-mechanism result on a synthetic TinyStories fixture**. It is not evidence of coding-repair capability, benchmark-wide reward validity, asynchronous speedup, or RSI. GO for further real-task verifier-consistency work; NO-GO for any model-capability or throughput claim from this pilot.

## Frozen experiment and audit

- Protocol: [`protocol.json`](protocol.json), SHA-256 `1e16f2452d2f0d101295df1438f87679bec621bb27f667259fe18d66db5d8523`.
- Model: `roneneldan/TinyStories-1M`, revision `77f1b168e219585646439073245fe87e56b3023e`; CPU fp32, one thread, at most three generated tokens.
- Each seed used three arms with 16 rollouts and 16 verifier calls per arm, eight-member GRPO batches, one actual AdamW update per trained arm, and four synthetic held-out prompts. The bounded async arm was serialized CPU execution; the queue overlapped only producer/verifier coroutines.
- Seeds 82106 and 82107 were both eligible under the frozen pilot continuation rule. Earlier aborted seeds and their outcome data remain excluded.
- Earlier independent audit and crash summary: [`pilot-audit-v2.json`](pilot-audit-v2.json) and [`sigkill-summary-v2.json`](sigkill-summary-v2.json). The original committed [`final-independent-audit-v6.json`](final-independent-audit-v6.json) remains unchanged at SHA-256 `fb5914e0f68cbcc815306eacbac709f0b0d40c92744bb73b05eed069a79768b8`. Review-driven auditor corrections produced v4 and v5 audits. The latest [`pilot-gate-audit-82106-correction-v6.json`](pilot-gate-audit-82106-correction-v6.json) and [`final-independent-audit-v6-correction-v6.json`](final-independent-audit-v6-correction-v6.json) use the output-safe matched-runner chain plus auditor correction v5. Primary outcomes and the decision exactly match the earlier correction-v5 report. Recovery audit rows now verify that eight SQLite trajectory states and their group-ack event agree with the worker summary, and that SQLite's active policy/checkpoint matches `active.json`, the journal, and checkpoint bytes. The full audit rechecked all five SIGKILL cutpoints; each has one committed transaction, eight consumed trajectories, and exact model/Adam/RNG/module-mode recovery. No inference or training was rerun. [`pilot-artifact-index-v5.json`](pilot-artifact-index-v5.json) and earlier audits remain immutable; the v6 audits and correction manifest are supplemental files.

## Two-seed verifier-drift results

| Seed | Stale group disagreements (false accepts, false rejects) | Naive invalid commits | VARE invalid commits | Stale-group handling |
|---|---:|---:|---:|---|
| 82106 | 2, 1 | 1/1 | 0/1 | VARE quarantined all 8 stale members; fresh group trained |
| 82107 | 3, 1 | 1/1 | 0/1 | VARE quarantined all 8 stale members; fresh group trained |

The prespecified primary contrast is invalid committed updates across the two seed bundles: naive `2/2` (Wilson 95% interval `[0.342, 1.000]`) versus VARE `0/2` (`[0.000, 0.658]`). The exact paired one-sided sign test is `p=0.25` (two non-tied pairs). This is a mechanism signal with severe uncertainty, not a statistically established general reliability gain. Under the same signed stale-v0 evaluation reports, the version-unaware negative-control gate accepted in both seeds and the strict expected-v1 gate rejected in both.

Each trained arm made exactly one optimizer call. Independent replay passed the frozen `2e-6` numerical tolerance; behavior-logprob alignment error was `1.57e-5`. Evaluation used only the four synthetic landing prompts and cannot support a coding-capability interpretation.

The synthetic held-out evaluator was saturated: every incumbent and candidate scored `4/4` in both seeds across all arms. The updates therefore showed no measured held-out quality gain, and this evaluator did not discriminate policies.

## Matched resource and latency observations

The following ranges are the two observed seed runs, not steady-state estimates. Per-arm throughput divides the arm's 16 generated rollouts and one useful group (zero for the invalid naive update) by its recorded `total_wall_s`, which includes runtime construction and checkpoint I/O. The model files were local; download/setup time is not included. Each arm ran once in fixed `sync`, `naive_async`, `vare_async` order in both seeds, and backend concurrency was one, so these rates are descriptive and do not establish a comparative async speedup.

| Arm | End-to-end arm wall | Rollouts/hour | Useful verified groups/hour | Committed updates/hour | Queue-inclusive group p50 / p95 | CPU core-equivalent |
|---|---:|---:|---:|---:|---:|---:|
| Sync | 4.92–5.77 s | 9,980–11,708 | 624–732 | 624–732 | 109.1–113.2 / 109.1–113.2 ms | 88.5–96.2% |
| Naive async | 0.98–1.11 s | 51,992–58,748 | 0 | 3,250–3,672, all invalid | 96.8–100.4 / 96.8–100.4 ms | 94.7–98.1% |
| VARE async | 2.75–2.83 s | 20,341–20,942 | 1,271–1,309 | 1,271–1,309 | 95.4–105.2 / 95.4–105.2 ms | 95.2–96.7% |

All arms used 16 verifier calls. VARE incurred zero extra verifier calls for safety and quarantined 8/16 rollouts (50%) as the stale mixed-version group. Per-seed process peak RSS was 1.28 GB (1.19 GiB) and measured swap growth was 0 bytes. The two checkpoints per arm totaled 56.8 MB (54.2 MiB); their combined write time was 0.109–0.146 s per arm. p50 and p95 coincide for each arm because the group-latency sample contains only two groups. Queue-wait p50 was 0.0208–0.0222 ms and p95 was 0.0233–0.0268 ms for the async arms. These small denominators and fixed arm order make the figures useful for accounting the pilot's safety cost, not for a performance ranking.

## Actual process-kill and recovery results

The five cutpoints ran serially as separate transactions against one held-fixed, independently verified eight-trajectory group. Each first process was confirmed dead after `SIGKILL`; each restart exited successfully, committed one effective update, consumed exactly eight trajectories, and passed the offline trajectory-ledger audit. Recovered model, Adam optimizer, CPU RNG, and module modes matched the frozen synchronous candidate exactly in all five cases.

| Cutpoint | Optimizer attempts across process lifetime | Effective commits | Recovery time | Peak RSS |
|---|---:|---:|---:|---:|
| Rollout complete, before admission | 1 | 1 | 4.32 s | 746 MB |
| Admission, before optimizer | 1 | 1 | 4.18 s | 724 MB |
| Optimizer computed, before commit | 2 | 1 | 4.16 s | 764 MB |
| Commit, before acknowledgment | 1 | 1 | 3.98 s | 694 MB |
| Evaluation, before promotion | 2 | 1 | 4.15 s | 761 MB |

Total harness time was 32.52 s. The two interrupted pre-commit attempts were recomputed, but no duplicate effective commit or consumed trajectory was observed. The five cases are not five independent tasks or rollout groups. No async throughput claim is made.

## Relation to asynchronous RL systems

The closest literature already studies asynchronous rollout/training overlap and policy staleness: [AReaL](https://arxiv.org/abs/2505.24298) decouples rollout and training with staleness-aware PPO; [StaleFlow](https://arxiv.org/abs/2601.12784) tracks trajectory lifecycle and controls staleness while addressing workload skew; [DORA](https://arxiv.org/abs/2604.26256) emphasizes multi-version rollout, data integrity, and bounded staleness; [PACE](https://arxiv.org/abs/2609.36830) separates generation and waiting staleness and uses pool-aware admission. Fault recovery is also established: [RobustRL](https://arxiv.org/abs/2512.22492) isolates and restores trainer/rollout roles in large GPU RL jobs. VARE therefore makes no claim to invent asynchronous RL, stale-trajectory filtering, data-integrity controls, or RL fault tolerance. This pilot's narrower tested intersection is semantic verifier-version provenance for rewards plus exact model/Adam/RNG recovery after learner-process termination. Two synthetic seeds, a single-host serialized backend, and one held-fixed crash group do not establish that this intersection is novel or superior; the appropriate outcome is an independently reproducible mechanism pilot pending a broader comparison.

The first crash attempt is retained and explicitly excluded: member-order training yielded a numerically close (`3.73e-9` maximum parameter difference) but non-identical model. The audit traced this to the crash harness omitting the frozen VARE replay-priority ordering and hook advantage path. The harness now applies `PrioritizedReplay.sample_current`, VARE sample conversion, and VARE group-relative advantages before the update. The frozen protocol, budgets, fixtures, and exact equality gate were not changed; the corrected run matches exactly. A later test-only correction makes the evaluator fixture skip cleanly when optional PyTorch is absent while keeping the replay-order regression runnable.

## Independent review corrections

Independent review found duplicate-cutpoint acceptance, NaN-masking tensor reducers, structured crash-inventory parsing gaps, a frozen-auditor reproduction defect, and a pilot-gate preflight that incorrectly required a crash bundle. These paths have focused regression coverage. A later P1 review found that a rejected output-path invocation could overwrite or add files in a retained run; the runner now writes failure evidence only after its own exclusive directory creation succeeds. Tests reproduce both the completed-run and existing-failure-record cases. The correction chain in [`runner-output-safety-correction-v1.json`](runner-output-safety-correction-v1.json), [`matched-runner-correction-v2.json`](matched-runner-correction-v2.json), and [`auditor-correction-v5.json`](auditor-correction-v5.json) binds the current runner, regression test and auditor provenance without changing the frozen protocol or measured outcomes. Three further review findings are closed by correction v5: behavior-logprob replay must meet the frozen `<1e-4` limit; consumed counts must match complete snapshot rows and acknowledgment heads; and the durable SQLite active pointer must match the worker journal, JSON pointer, and checkpoint digest. Corruption regressions fail on stale pre-ack snapshots, mismatched acknowledgment heads, a stale database active pointer, and threshold/non-finite behavior errors. The auditor still requires the exact five-cutpoint set, rejects non-finite tensor differences, validates artifact inventories, and supports pilot-only audits without crash bundles.

Two distinct failed audit observations during this correction are preserved in `results/verifier_consistent_async_grpo_v6/audit-integrity-review-postmortem-v2.json`. The first exposed the structured-manifest parser gap; the second exposed SQLite's disposable sidecars in the exact inventory check. The final v6 audit passes after handling those cases explicitly and rejecting noncanonical paths, symlinks, and unmanifested temporary files.

## Reproduction

Use the pinned local TinyStories and RVL checkouts named in the protocol. Run with `python -S` so site startup hooks cannot change source imports. The measured matched-runner source remains frozen at commit `69e9ab9b22da16413dbeababa0552e36aa06612f`. Seeds 82106 and 82107 are consumed and must not be regenerated as new evidence. The commands below audit retained outputs only; they do not run model inference or training. Use new audit output paths and preserve the source bundles.

```sh
python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v5.json \
  --runner-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/matched-runner-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/verifier_consistent_async_grpo_v6/offline-audit-pilot-correction-v6.json

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --run results/verifier_consistent_async_grpo_v6/seed-82107 \
  --crash-run results/verifier_consistent_async_grpo_v6/crash-82106-replay-order-v2 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v5.json \
  --runner-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/matched-runner-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/verifier_consistent_async_grpo_v6/offline-audit-full-correction-v6.json
```

To verify retained outputs, check each run's `SHA256.json` against its files. The failed first crash attempt, all aborted seeds, and their manifests remain preserved and excluded.
