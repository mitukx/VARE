from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT.parent / "sha256-manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_event_chain(path: Path) -> tuple[int, str]:
    previous = "0" * 64
    count = 0
    for expected_sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        event = row["event"]
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        current = hashlib.sha256(canonical).hexdigest()
        assert event.get("sequence") == expected_sequence, (path, expected_sequence)
        assert row.get("previous_event_sha256") == previous, (path, expected_sequence, "link")
        assert row.get("event_sha256") == current, (path, expected_sequence, "event hash")
        previous = current
        count += 1
    assert count, path
    return count, previous


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT.parent / entry["path"]
        assert path.is_file(), path
        assert path.stat().st_size == entry["bytes"], path
        assert digest(path) == entry["sha256"], path

    for arm in ("01-A_autonomous-seed-2029", "02-B_localized-seed-2029"):
        attempt = ROOT / "attempts" / arm
        rows, last_hash = verify_event_chain(attempt / "events.jsonl")
        assessment = json.loads((attempt / "attempt-assessment-v2.json").read_text())
        assert assessment["event_chain_valid"] is True
        assert assessment["event_count"] == rows
        assert assessment["event_chain_final_sha256"] == last_hash
        assert assessment["hidden_grader"] == "NOT_RUN_NO_CANDIDATE_PATCH"
        assert assessment["verified_repair"] is False
        assert assessment["candidate_source_unchanged"] is True
        assert assessment["canonical_target_diff_bytes"] == 0

    qualification = json.loads((ROOT / "qualification-summary.json").read_text())
    assert qualification["fresh_development_family_eligible"] is False
    assert qualification["heldout_eligible"] is False
    exposure = json.loads((ROOT / "append-only-prior-exposure-reconciliation-v1.json").read_text())
    assert len(exposure["prior_vare_model_attempts"]) == 2
    assert "not fresh" in exposure["finding"].lower()
    print(f"verified {len(manifest['files'])} bundle files, both event chains, and prior-exposure disqualification")


if __name__ == "__main__":
    main()
