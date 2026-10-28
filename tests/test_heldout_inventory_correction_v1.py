"""Pin the append-only prospective held-out inventory correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/heldout-inventory-correction-v1"
RECORD = DIRECTORY / "heldout-inventory-correction-v1.json"
SIDECAR = DIRECTORY / "heldout-inventory-correction-v1.sha256"
RECORD_SHA256 = "7fc013e942cf7686059a8d3d1bfd044758caf543ad34e1eb7a412577810890c2"
REMAINING = {
    "transformers-48319-stale-best-checkpoint",
    "transformers-49208-resume-metrics",
    "transformers-49147-gemma3n-fp16-gradient-clipping",
    "mlxlm-1921-padded-logprobs",
    "mlxlm-1938-phi3small-logits-mask",
    "mlxlm-1979-ssm-padding-mask",
    "mlxlm-1898-qwen3-rope-parameters",
    "mlxlm-1877-compressed-tensors-format-validation",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_correction_record_bytes_and_inventory_decision():
    blob = RECORD.read_bytes()
    assert _sha256(RECORD) == RECORD_SHA256
    digest, name = SIDECAR.read_text(encoding="utf-8").split()
    assert digest == RECORD_SHA256
    assert name == "results/heldout-inventory-correction-v1/heldout-inventory-correction-v1.json"
    payload = json.loads(blob.decode("utf-8"))
    assert payload["schema"] == "vare.heldout-inventory-correction.v1"
    assert payload["append_only"] is True
    assert payload["historical_records_modified"] is False
    assert payload["verification"]["identity_1880_versus_gptq_groups"] == "CONFIRMED"
    assert payload["verification"]["exposure_of_gptq_groups"] == "CONFIRMED"
    excluded = payload["excluded_task"]
    assert excluded["upstream_pr"] == 1880
    assert excluded["heldout_task_id"] == "mlxlm-1880-gptq-group-error-propagation"
    assert excluded["development_task_id"] == "gptq-groups"
    assert excluded["historical_certificate_retained"] is True
    inventory = payload["prospective_inventory"]
    assert inventory["current_source_grader_certified_unique_families"] == 8
    assert inventory["required"] == 15
    assert inventory["count_expression"] == "8/15"
    assert inventory["gate_status"] == "NOT_MET"
    assert inventory["fifteen_task_gate_met"] is False
    assert inventory["fully_model_eligible_heldout_total"] == "UNKNOWN_NOT_ESTABLISHED"
    assert inventory["heldout_model_inference_admitted"] is False
    assert set(inventory["remaining_source_grader_certified_task_ids"]) == REMAINING
    assert len(inventory["remaining_tasks"]) == 8
    statement = payload["non_exposure_statement"]
    assert statement.startswith(
        "A VARE-level finding of no exposure match is not evidence that a pretrained model never saw the task."
    )
    open_items = payload["open_unverified_inconsistencies"]
    assert [item["id"] for item in open_items] == [
        "readiness_hash_mismatch",
        "1926_fix_commit_discrepancy",
        "49412_split_conflict",
        "snapshot_or_theme_overlaps",
    ]
    assert all(item["status"] == "OPEN_UNVERIFIED" for item in open_items)
    assert all(item["decided_disqualification"] is False for item in open_items)
    superseded = {(item["path"], item["sha256"]) for item in payload["supersedes"]}
    assert superseded == {
        (
            "results/sprint_iv_20261027/exposure-disposition-append-v1.json",
            "3018268f0e14ccf75e52234e1f84aa4a90db10cdb993b3f897870b288cdfd66c",
        ),
        (
            "results/sprint_iv_20261027/exposure-disposition-append-v2.json",
            "b2da5a3d7ee4e09b0d0cae890b4fbf3d1edc2a5a950c5ec3a7811b80c51b3ed5",
        ),
    }
    disagreements = {item["item"] for item in payload["verification"]["disagreements_with_20261028_rederivation"]}
    assert disagreements == {"attempt_commit_attribution", "consumption_sentence_path"}
