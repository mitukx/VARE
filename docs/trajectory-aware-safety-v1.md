# TASE v1 — Trajectory-Aware Safety Evaluator (bounded reference)

TASE extends outcome-only evaluation with an independently collected process audit. It is opt-in and additive. It does not replace the existing `SafetyInterlock`, `PromotionGate`, `EvidenceTrustRoot`, promotion journal, or RVL trajectory contract.

## Scope and trust boundaries

- `src/vare/trajectory_safety.py` defines an event schema, frozen policy allowlists, a `HashChainLedger`-backed recorder, a strict closed-trace parser and a deterministic outcome (pass / violation / unknown). A denied action is still a violation even when task outcome is correct.
- The recorder needs a runtime adapter outside the candidate privilege domain. Every tool dispatch, network attempt, and artifact event included in the claimed audit scope must be intercepted independently. The dispatch counter passed to `close()` must also be independent. A candidate-supplied trace is untrusted.
- `src/vare/trajectory_dispatch.py` adds a cooperative trusted dispatcher that journals tool and outbound HTTP attempts before a side effect and records returned/error outcomes. Tools must be registered in the frozen allowlist; hostnames use exact matching. Denied calls are retained as safety violations. Redirects must be mediated hop-by-hop by the HTTP adapter, which is not implemented here. The dispatcher now requires an explicit `close(expected_event_count=...)` from the independent runtime dispatch counter; its own recorded count is never accepted as a coverage oracle. This catches count mismatches but not uninstrumented actions absent from both counters. The caller supplies a trusted async execution function; this does not intercept arbitrary process I/O.
- `src/vare/trajectory_provenance.py` replays explicit single-parent event dependencies after validating the ledger, and identifies operations with recorded ancestry from external webpages/tool outputs. This is observed lineage, not inferred semantic causation, a general DAG, or grounds to prohibit an otherwise authorized operation.
- `src/vare/trajectory_promotion.py` validates a signed `EvaluationReport` carrying `metadata["trajectory_safety"]` with the frozen policy SHA-256. Configure `PromotionConfig.trajectory_policy_sha256` to activate fail-closed gating. It applies to both incumbent and candidate evaluation reports; unset configuration keeps historical behavior.
- A trusted evaluator should call `audit_trajectory(...).to_metadata()`, attach it to its report as `trajectory_safety`, and sign the complete report using the existing checkpoint-bound `EvidenceTrustRoot.attest` path. Never give the candidate the signing key.
- Hash chaining detects changes to stored bytes, but does not ensure that events were collected in the first place, prevent a privileged operator from replacing an unanchored chain, or guarantee OS isolation. The audit code does not intercept network traffic or establish a hostile-code sandbox.
- This initial pass is only event-level authorization and exact egress-host filtering. It does **not** implement data-flow taint propagation, payload exfiltration analysis, actual credential-use instrumentation, graph-level semantic causality, or secure TLS-level redirect resolution. Those require independent runtime instrumentation and explicit experiments before any claim.
- `unknown`, missing closure, count mismatch, broken hash chain, parent gaps, tampering, wrong run/policy binding, or absent/unauthenticated metadata cannot authorize promotion. A trusted signer must validate raw trace evidence before attestation; the signature alone is not a proof of the truth of its inputs.

## Capture completion

The dispatcher refuses `close()` while an owned call is pending. Capture persistence failures fence new calls and closure; keep the partial trace and reconcile side effects before creating a fresh run. Tool callbacks must own actual worker completion rather than detach work. Dispatch and offline replay use the same URL predicate, including port, fragment, whitespace and exact hostname checks. See the [measured capture-boundary correction](trajectory-capture-boundary-v1-report.md).

## Quick verification

```bash
python -m pytest tests/test_trajectory_safety.py tests/test_trajectory_dispatch.py tests/test_trajectory_promotion.py tests/test_trajectory_provenance.py tests/test_promotion.py
```

## Follow-up research, not current evidence

Freeze a protocol before experiments: compare a fixed safety evaluator, scheduled refresh, and policy-shift-triggered refresh under matched audits and compute. Measure unsafe promotion rate with confidence intervals, conditional miss rate on malicious trajectories, false positive rate, detector latency, task success, and overhead. Distinguish engineered examples from actual model-agent trajectories. Do not report simulated outcomes as real-model improvement or protection of a hostile runtime.
