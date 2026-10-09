# Promotion evidence identity v1

## Question

Can a candidate pass the promotion gate while omitting a reported slice, or can the engine promote a policy based on an evaluation report labeled as another policy?

The frozen CPU protocol and hash lock are [`promotion_evidence_identity_v1.json`](../protocols/promotion_evidence_identity_v1.json) and [`promotion_evidence_identity_v1.lock.json`](../protocols/promotion_evidence_identity_v1.lock.json). This is a deterministic software contract experiment, not a model or policy-quality measurement.

## Baseline

At pre-fix commit `6ad75dba16dc8dc34ede85790ec86820591f235a`, a candidate report containing only the `easy` slice was accepted even though the incumbent also reported `hard`. In the engine integration case, the trainer produced `p1`, but the candidate evaluation hook returned a favorable report labeled `p0`; the engine promoted `p1`. The positive control with matching identities and complete slice coverage was also accepted. Raw results are in [`baseline.json`](../results/promotion-evidence-identity-v1/baseline.json).

## Fix and result

`PromotionGate` now rejects different slice-key sets with `slice_coverage_mismatch`. `CapabilityLoop` checks that each report's `policy_id` matches the ID passed to its evaluation hook; a mismatch yields a rejected decision and discards the candidate. The unchanged reproducer now rejects both malformed cases and still promotes the matching control; see [`fixed.json`](../results/promotion-evidence-identity-v1/fixed.json).

## Limits

This verifies identity and coverage declarations only. It cannot prove that an evaluation backend actually loaded the named policy, used held-out data, or measured a meaningful capability. One constructed contract test gives no estimate of production error prevalence and no learning evidence. GitHub Actions run [37857916437](https://github.com/mitukx/VARE/actions/runs/37857916437) passed 187 tests with 12 skipped, plus both exact GRPO audit recomputations and the demo; see the [CI record](../results/promotion-evidence-identity-v1/ci.json).
