"""Opt-in promotion evidence MACs for an externally isolated trusted evaluator.

This is integrity/authenticity between *separate privilege domains*, not a
sandbox or proof that the grader is independent/correct. Never expose the MAC
key to candidate processes, model tools, or candidate workspaces.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import re
from typing import Any

from .evidence import canonical_json
from .types import EvaluationReport

_RECEIPT = "vare_evidence_receipt"
_DOMAIN = "vare.promotion-evidence.v1"


@dataclass(frozen=True, repr=False, slots=True)
class EvidenceTrustRoot:
    """Pinned, operator-owned evaluation identity and shared-secret verifier.

    Create a fresh, unpredictable run_id for each independent promotion study.
    The trusted evaluator signs with this root; the promotion process only
    verifies. In a real deployment the processes must be separated.
    """

    key: bytes
    protocol_sha256: str
    evaluator_sha256: str
    run_id: str

    def __post_init__(self) -> None:
        if type(self.key) is not bytes or len(self.key) < 32:
            raise ValueError("evidence key must contain at least 32 bytes")
        for field_name in ("protocol_sha256", "evaluator_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not isinstance(self.run_id, str) or not 16 <= len(self.run_id) <= 256:
            raise ValueError("run_id must be a nonempty, operator-issued run identifier")

    @staticmethod
    def _report_body(report: EvaluationReport) -> dict[str, Any]:
        if not isinstance(report, EvaluationReport) or not isinstance(report.metadata, dict):
            raise ValueError("report metadata must be a dictionary")
        if not isinstance(report.slices, dict):
            raise ValueError("report slices must be a dictionary")
        return {
            "policy_id": report.policy_id,
            "primary": report.primary,
            "slices": report.slices,
            "cost": report.cost,
            "verifier_disagreement": report.verifier_disagreement,
            "n": report.n,
            "metadata": {k: v for k, v in report.metadata.items() if k != _RECEIPT},
        }

    def _mac(self, report: EvaluationReport) -> str:
        envelope = {
            "domain": _DOMAIN,
            "protocol_sha256": self.protocol_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "run_id": self.run_id,
            "report": self._report_body(report),
        }
        payload = canonical_json(envelope).encode("utf-8")
        return hmac.new(self.key, payload, hashlib.sha256).hexdigest()

    def attest(self, report: EvaluationReport) -> EvaluationReport:
        """Call ONLY from a trusted grader after scoring independently."""
        if _RECEIPT in report.metadata:
            raise ValueError("cannot attest a report that already has a receipt")
        mac = self._mac(report)
        return replace(
            report,
            slices=dict(report.slices),
            metadata={**report.metadata, _RECEIPT: {"schema": _DOMAIN, "mac_sha256": mac}},
        )

    def verifies(self, report: EvaluationReport) -> bool:
        try:
            receipt = report.metadata.get(_RECEIPT)
            if (
                type(receipt) is not dict
                or set(receipt) != {"schema", "mac_sha256"}
                or receipt["schema"] != _DOMAIN
                or not isinstance(receipt["mac_sha256"], str)
            ):
                return False
            return hmac.compare_digest(receipt["mac_sha256"], self._mac(report))
        except (ValueError, TypeError, OverflowError, RecursionError, AttributeError):
            return False
