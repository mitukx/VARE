# V6 verifier-consistent GRPO and process-recovery results

## Result and decision

The preregistered two-seed synthetic verifier-drift pilot completed, followed by five actual learner-process `SIGKILL` injections. The matched VARE path rejected both stale groups and committed only fresh, verifier-consistent updates. The naive path committed one invalid mixed-label update in each seed. Every crash case recovered to the exact synchronous model, Adam, CPU RNG, and module-mode state, with one effective commit and no duplicate trajectory consumption.

This is a **systems-mechanism result on a synthetic TinyStories fixture**. It is not evidence of coding-repair capability, benchmark-wide reward validity, asynchronous speedup, or RSI. GO for further real-task verifier-consistency work; NO-GO for any model-capability or throughput claim from this pilot.

## Frozen experiment and audit

- Protocol: [`protocol.json`](protocol.json), SHA-256 `1e16f2452d2f0d101295df1438f87679bec621bb27f667259fe18d66db5d8523`.
- Model: `roneneldan/TinyStories-1M`, revision `77f1b168e219585646439073245fe87e56b3023e`; CPU fp32, one thread, at most three generated tokens.
- Each seed used three arms with 16 rollouts and 16 verifier calls per arm, eight-member GRPO batches, one actual AdamW update per trained arm, and four synthetic held-out prompts. The bounded async arm was serialized CPU execution; the queue overlapped only producer/verifier coroutines.
- Seeds 82106 and 82107 were both eligible under the frozen pilot continuation rule. Earlier aborted seeds and their outcome data remain excluded.
- Independent audit: [`pilot-audit-v2.json`](pilot-audit-v2.json). The corrected auditor verifies the frozen source snapshots and independently replays every model update. The five-cutpoint machine-readable summary is [`sigkill-summary-v2.json`](sigkill-summary-v2.json); [`pilot-artifact-index-v2.json`](pilot-artifact-index-v2.json) links these to the full raw runs and their SHA-256 inventories.

## Two-seed verifier-drift results

| Seed | Stale group disagreements (false accepts, false rejects) | Naive invalid commits | VARE invalid commits | Stale-group handling |
|---|---:|---:|---:|---|
| 82106 | 2, 1 | 1/1 | 0/1 | VARE quarantined all 8 stale members; fresh group trained |
| 82107 | 3, 1 | 1/1 | 0/1 | VARE quarantined all 8 stale members; fresh group trained |

The prespecified primary contrast is invalid committed updates across the two seed bundles: naive `2/2` (Wilson 95% interval `[0.342, 1.000]`) versus VARE `0/2` (`[0.000, 0.658]`). The exact paired one-sided sign test is `p=0.25` (two non-tied pairs). This is a mechanism signal with severe uncertainty, not a statistically established general reliability gain. Under the same signed stale-v0 evaluation reports, the version-unaware negative-control gate accepted in both seeds and the strict expected-v1 gate rejected in both.

Each trained arm made exactly one optimizer call. Independent replay passed the frozen `2e-6` numerical tolerance; behavior-logprob alignment error was `1.57e-5`. Evaluation used only the four synthetic landing prompts and cannot support a coding-capability interpretation.

The synthetic held-out evaluator was saturated: every incumbent and candidate scored `4/4` in both seeds across all arms. The updates therefore showed no measured held-out quality gain, and this evaluator did not discriminate policies.

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

The first crash attempt is retained and explicitly excluded: member-order training yielded a numerically close (`3.73e-9` maximum parameter difference) but non-identical model. The audit traced this to the crash harness omitting the frozen VARE replay-priority ordering and hook advantage path. The harness now applies `PrioritizedReplay.sample_current`, VARE sample conversion, and VARE group-relative advantages before the update. The frozen protocol, budgets, fixtures, and exact equality gate were not changed; the corrected run matches exactly. A later test-only correction makes the evaluator fixture skip cleanly when optional PyTorch is absent while keeping the replay-order regression runnable.

## Reproduction

Use the pinned local TinyStories and RVL checkouts named in the protocol. Run with `python -S` so site startup hooks cannot change source imports.

```sh
python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82106 \
  --output results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/benchmark_verifier_consistent_async_grpo.py \
  --seed 82107 \
  --output results/verifier_consistent_async_grpo_v6/seed-82107 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned

python -S scripts/benchmark_verifier_consistent_grpo_sigkill.py \
  --source-run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --seed 82106 \
  --output results/verifier_consistent_async_grpo_v6/crash-82106-replay-order-v2 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v2.json

python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --run results/verifier_consistent_async_grpo_v6/seed-82107 \
  --crash-run results/verifier_consistent_async_grpo_v6/crash-82106-replay-order-v2 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/verifier_consistent_async_grpo_v6/audit-with-crash-replay-order-v2.json
```

Do not overwrite an existing run directory. To verify retained outputs, check each run's `SHA256.json` against its files. The failed first crash attempt, all aborted seeds, and their manifests remain preserved and excluded.
