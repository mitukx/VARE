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
_BOUND_DOMAIN = "vare.promotion-evidence.checkpoint.v2"
_CHECKPOINT = "vare_checkpoint_sha256"
_EVAL_SET = "vare_evaluation_set_sha256"
_NONCE = "vare_promotion_nonce"


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
    evaluation_set_sha256: str | None = None
    promotion_nonce: str | None = None
    strict_checkpoint_binding: bool = False

    def __post_init__(self) -> None:
        if type(self.key) is not bytes or len(self.key) < 32:
            raise ValueError("evidence key must contain at least 32 bytes")
        for field_name in ("protocol_sha256", "evaluator_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if not isinstance(self.run_id, str) or not 16 <= len(self.run_id) <= 256:
            raise ValueError("run_id must be a nonempty, operator-issued run identifier")
        if type(self.strict_checkpoint_binding) is not bool:
            raise ValueError("strict_checkpoint_binding must be bool")
        if self.strict_checkpoint_binding:
            if (
                not isinstance(self.evaluation_set_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.evaluation_set_sha256) is None
                or not isinstance(self.promotion_nonce, str)
                or not 16 <= len(self.promotion_nonce) <= 256
            ):
                raise ValueError("strict evidence needs pinned dataset SHA-256 and unique promotion nonce")
        elif self.evaluation_set_sha256 is not None or self.promotion_nonce is not None:
            raise ValueError("binding fields require strict_checkpoint_binding")

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

    def _binding_matches(self, report: EvaluationReport) -> bool:
        if not self.strict_checkpoint_binding:
            return True
        if not isinstance(report, EvaluationReport) or not isinstance(report.metadata, dict):
            return False
        digest = report.metadata.get(_CHECKPOINT)
        return (
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            and report.metadata.get(_EVAL_SET) == self.evaluation_set_sha256
            and report.metadata.get(_NONCE) == self.promotion_nonce
        )

    def checkpoint_digest(self, report: EvaluationReport) -> str | None:
        """Return the signed checkpoint digest only after verifying the receipt."""
        if not self.strict_checkpoint_binding or not self.verifies(report):
            return None
        digest = report.metadata[_CHECKPOINT]
        return digest if isinstance(digest, str) else None

    def _mac(self, report: EvaluationReport) -> str:
        envelope = {
            "domain": _BOUND_DOMAIN if self.strict_checkpoint_binding else _DOMAIN,
            "protocol_sha256": self.protocol_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "run_id": self.run_id,
            "report": self._report_body(report),
        }
        payload = canonical_json(envelope).encode("utf-8")
        return hmac.new(self.key, payload, hashlib.sha256).hexdigest()

    def attest(self, report: EvaluationReport, *, checkpoint_sha256: str | None = None) -> EvaluationReport:
        """Trusted evaluator ONLY: hash the actual evaluated checkpoint independently."""
        if not isinstance(report, EvaluationReport) or not isinstance(report.metadata, dict):
            raise ValueError("report must include valid metadata")
        if _RECEIPT in report.metadata:
            raise ValueError("cannot attest a report that already has a receipt")
        if self.strict_checkpoint_binding:
            if (
                not isinstance(checkpoint_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None
                or any(k in report.metadata for k in (_CHECKPOINT, _EVAL_SET, _NONCE))
            ):
                raise ValueError("checkpoint digest missing/invalid or binding already supplied")
            metadata = {
                **report.metadata,
                _CHECKPOINT: checkpoint_sha256,
                _EVAL_SET: self.evaluation_set_sha256,
                _NONCE: self.promotion_nonce,
            }
        else:
            if checkpoint_sha256 is not None:
                raise ValueError("checkpoint binding requires strict mode")
            metadata = dict(report.metadata)
        unsigned = replace(report, slices=dict(report.slices), metadata=metadata)
        mac = self._mac(unsigned)
        schema = _BOUND_DOMAIN if self.strict_checkpoint_binding else _DOMAIN
        return replace(
            unsigned, metadata={**unsigned.metadata, _RECEIPT: {"schema": schema, "mac_sha256": mac}}
        )

    def verifies(self, report: EvaluationReport) -> bool:
        try:
            receipt = report.metadata.get(_RECEIPT)
            if (
                type(receipt) is not dict
                or set(receipt) != {"schema", "mac_sha256"}
                or receipt["schema"] != (_BOUND_DOMAIN if self.strict_checkpoint_binding else _DOMAIN)
                or not isinstance(receipt["mac_sha256"], str)
                or not self._binding_matches(report)
            ):
                return False
            return hmac.compare_digest(receipt["mac_sha256"], self._mac(report))
        except (ValueError, TypeError, OverflowError, RecursionError, AttributeError):
            return False
