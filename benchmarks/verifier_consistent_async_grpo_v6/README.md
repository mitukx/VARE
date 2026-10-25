# Verifier-consistent asynchronous GRPO v6

The v6 study is complete. Seeds 82106 and 82107 are consumed; do not rerun them as new evidence. The frozen protocol remains [`protocol.json`](protocol.json), SHA-256 `1e16f2452d2f0d101295df1438f87679bec621bb27f667259fe18d66db5d8523`. The measured runner and source inventory are frozen at commit `69e9ab9b22da16413dbeababa0552e36aa06612f`.

This was a synthetic TinyStories verifier-drift and single-host recovery pilot, not a coding-capability test or evidence of general asynchronous speedup. The [results report](crash-recovery-results-v6.md) gives the complete outcomes and limits. Earlier failed attempts and correction manifests remain preserved.

## Offline audit of retained runs

The runner output-safety correction changes no experiment rule. Current source corrections are bound by [`runner-output-safety-correction-v1.json`](runner-output-safety-correction-v1.json), [`matched-runner-correction-v2.json`](matched-runner-correction-v2.json), [`auditor-correction-v4-runner-v2.json`](auditor-correction-v4-runner-v2.json), and [`auditor-correction-v5.json`](auditor-correction-v5.json). Auditor correction v5 chains v4 and additionally enforces the frozen behavior-logprob threshold, compares claimed trajectory consumption with snapshot rows and acknowledgment events, and binds the durable active row to the journal, JSON pointer, and recovered checkpoint bytes. The original frozen protocol and earlier correction manifests are unchanged.

Use `python -S` and the already pinned local model and RVL source directories. These commands audit retained bundles only; they do not run inference or training. Each output path must be new.

Pilot gate:

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
```

Full two-seed and five-cutpoint audit:

```sh
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

The retained [`pilot-gate-audit-82106-correction-v6.json`](pilot-gate-audit-82106-correction-v6.json) and [`final-independent-audit-v6-correction-v6.json`](final-independent-audit-v6-correction-v6.json) are reruns under correction v5. The full audit's primary metrics and decision exactly match correction v5; its recovery records additionally expose the independently queried SQLite consumption and promotion rows. Earlier audit reports and [`pilot-artifact-index-v5.json`](pilot-artifact-index-v5.json) remain immutable; the supplemental audits do not modify raw-run artifacts. The correction-v6 output paths above are for independent reruns and must be new when used.
