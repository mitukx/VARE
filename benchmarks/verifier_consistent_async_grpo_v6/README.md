# Verifier-consistent asynchronous GRPO v6

The v6 study is complete. Seeds 82106 and 82107 are consumed; do not rerun them as new evidence. The frozen protocol remains [`protocol.json`](protocol.json), SHA-256 `1e16f2452d2f0d101295df1438f87679bec621bb27f667259fe18d66db5d8523`. The measured runner and source inventory are frozen at commit `69e9ab9b22da16413dbeababa0552e36aa06612f`.

This was a synthetic TinyStories verifier-drift and single-host recovery pilot, not a coding-capability test or evidence of general asynchronous speedup. The [results report](crash-recovery-results-v6.md) gives the complete outcomes and limits. Earlier failed attempts and correction manifests remain preserved.

## Offline audit of retained runs

The runner output-safety correction changes no experiment rule. Current source corrections are bound by [`runner-output-safety-correction-v1.json`](runner-output-safety-correction-v1.json), [`matched-runner-correction-v2.json`](matched-runner-correction-v2.json), and [`auditor-correction-v4-runner-v2.json`](auditor-correction-v4-runner-v2.json). The latter binds the exact runner-manifest hash, while the runner manifest chains the earlier non-finite replay correction. The original v1/v4 manifests and frozen protocol are unchanged.

Use `python -S` and the already pinned local model and RVL source directories. These commands audit retained bundles only; they do not run inference or training. Each output path must be new.

Pilot gate:

```sh
python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --allow-partial \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v4-runner-v2.json \
  --runner-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/matched-runner-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/verifier_consistent_async_grpo_v6/offline-audit-pilot-v5.json
```

Full two-seed and five-cutpoint audit:

```sh
python -S scripts/audit_verifier_consistent_async_grpo.py \
  --run results/verifier_consistent_async_grpo_v6/seed-82106 \
  --run results/verifier_consistent_async_grpo_v6/seed-82107 \
  --crash-run results/verifier_consistent_async_grpo_v6/crash-82106-replay-order-v2 \
  --model-dir ../tinystories-1m-pinned \
  --rvl-source ../rvl-pinned \
  --auditor-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/auditor-correction-v4-runner-v2.json \
  --runner-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/matched-runner-correction-v2.json \
  --crash-correction-manifest benchmarks/verifier_consistent_async_grpo_v6/crash-runner-correction-v3.json \
  --output results/verifier_consistent_async_grpo_v6/offline-audit-full-v5.json
```

The resulting [`pilot-gate-audit-82106-correction-v5.json`](pilot-gate-audit-82106-correction-v5.json) and [`final-independent-audit-v6-correction-v5.json`](final-independent-audit-v6-correction-v5.json) are retained supplemental audits. Their measured outcomes exactly match the v4 audit after excluding the auditor- and runner-correction-manifest hashes. The older [`pilot-artifact-index-v5.json`](pilot-artifact-index-v5.json) remains an immutable snapshot of its earlier inventory; these later audits and manifests supplement it without changing raw-run artifacts.
