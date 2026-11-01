from scripts.verify_vare_ig_evidence import verify_evidence


def test_committed_vare_ig_research_evidence_is_hash_consistent():
    result = verify_evidence()
    assert result["evidence_files"] >= 100
    assert result["protocol_files"] >= 10
    assert result["run_source_files"] == 14
    assert result["trace_events"] == 162
