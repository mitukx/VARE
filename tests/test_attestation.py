"""CPU-only malicious evidence fixtures for authenticated candidate promotion."""
from dataclasses import replace
import hashlib
import secrets

from vare.attestation import EvidenceTrustRoot
from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


def root(*, run_id="promotion-study-run-2026-10-14", evaluator=b"grader-v1"):
    return EvidenceTrustRoot(
        key=secrets.token_bytes(32),
        protocol_sha256=hashlib.sha256(b"locked-test-protocol").hexdigest(),
        evaluator_sha256=hashlib.sha256(evaluator).hexdigest(),
        run_id=run_id,
    )


def pair(trust):
    incumbent = trust.attest(EvaluationReport("old", 0.5, {"hard": 0.4}, n=10))
    candidate = trust.attest(EvaluationReport("new", 0.8, {"hard": 0.7}, n=10))
    return incumbent, candidate


def gate(trust):
    return PromotionGate(
        PromotionConfig(min_primary_gain=0.01, min_eval_examples=10),
        trust_root=trust,
    )


def rejected(trust, incumbent, candidate):
    decision = gate(trust).decide(incumbent, candidate)
    assert not decision.accepted
    assert decision.reasons == ("untrusted_evaluation_evidence",)


def test_valid_attested_pair_passes_but_unsigned_candidate_fails():
    trust = root()
    incumbent, candidate = pair(trust)
    assert gate(trust).decide(incumbent, candidate).accepted
    rejected(trust, incumbent, EvaluationReport("new", 0.99, {"hard": 0.9}, n=10))


def test_score_slice_identity_and_count_tampering_fails_closed():
    trust = root()
    incumbent, candidate = pair(trust)
    for changed in (
        replace(candidate, primary=0.99),
        replace(candidate, slices={"hard": 0.99}),
        replace(candidate, policy_id="different-model"),
        replace(candidate, n=10000),
        replace(candidate, metadata={**candidate.metadata, "extra": "changed"}),
    ):
        rejected(trust, incumbent, changed)


def test_wrong_evaluator_protocol_or_run_cannot_replay():
    trust = root()
    incumbent, candidate = pair(trust)
    for other in (
        EvidenceTrustRoot(trust.key, trust.protocol_sha256,
                          hashlib.sha256(b"grader-v2").hexdigest(), trust.run_id),
        EvidenceTrustRoot(trust.key, hashlib.sha256(b"other-protocol").hexdigest(),
                          trust.evaluator_sha256, trust.run_id),
        EvidenceTrustRoot(trust.key, trust.protocol_sha256,
                          trust.evaluator_sha256, "second-experiment-unique"),
        EvidenceTrustRoot(secrets.token_bytes(32), trust.protocol_sha256,
                          trust.evaluator_sha256, trust.run_id),
    ):
        rejected(other, incumbent, candidate)


def test_malformed_receipt_or_data_never_authorizes_promotion():
    trust = root()
    incumbent, candidate = pair(trust)
    for receipt in (None, "", {"schema": "unknown", "mac_sha256": "a" * 64},
                    {"schema": "vare.promotion-evidence.v1", "mac_sha256": None}):
        forged = replace(candidate, metadata={**candidate.metadata, "vare_evidence_receipt": receipt})
        rejected(trust, incumbent, forged)


def test_cannot_launder_an_existing_receipt_and_config_requires_secret():
    trust = root()
    _, candidate = pair(trust)
    import pytest
    with pytest.raises(ValueError, match="already has a receipt"):
        trust.attest(candidate)
    with pytest.raises(ValueError, match="32 bytes"):
        EvidenceTrustRoot(b"weak", trust.protocol_sha256, trust.evaluator_sha256, trust.run_id)
