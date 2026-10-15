"""Opt-in promotion constraint for independently attested trajectory audits."""
from __future__ import annotations

from dataclasses import dataclass
import re
from .attestation import EvidenceTrustRoot
from .types import EvaluationReport

_SCHEMA = "vare.trajectory-safety.v1"


@dataclass(frozen=True, slots=True)
class TrajectoryPromotionConstraint:
    """Require that independent evaluator receipts bind an eligible safety audit.

    The trust root must be owned by an evaluator isolated from the candidate;
    an unsigned candidate-supplied metadata dictionary is never acceptable.
    """
    trust_root: EvidenceTrustRoot
    required_policy_sha256: str

    def reason(self, report: EvaluationReport) -> str | None:
        if not self.trust_root.verifies(report):
            return "untrusted_trajectory_evidence"
        audit = report.metadata.get("trajectory_safety")
        if not isinstance(audit, dict):
            return "missing_trajectory_evidence"
        if (
            audit.get("schema") != _SCHEMA
            or audit.get("policy_id") != report.policy_id
            or audit.get("policy_sha256") != self.required_policy_sha256
            or not isinstance(audit.get("run_id"), str)
            or audit["run_id"] != self.trust_root.run_id
            or not isinstance(audit.get("ledger_head_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", audit["ledger_head_sha256"]) is None
            or type(audit.get("event_count")) is not int
            or type(audit.get("expected_event_count")) is not int
            or audit["event_count"] < 0
            or audit["expected_event_count"] != audit["event_count"]
            or audit.get("capture_complete") is not True
        ):
            return "invalid_trajectory_evidence"
        if audit.get("status") != "pass" or audit.get("violations") != []:
            return "unsafe_or_unverified_trajectory"
        return None
