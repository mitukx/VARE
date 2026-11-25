from __future__ import annotations

import os

from scripts.sprint_viii_synthetic_grader_v1 import grade_bytes, read_candidate


def test_semantic_grader_calibration_baseline_known_fix_and_wrong_mutant() -> None:
    assert grade_bytes(b"def shift(value):\n    return value\n")["status"] == "FAIL"
    assert grade_bytes(b"def shift(value):\n    return value + 2\n")["status"] == "PASS"
    assert grade_bytes(b"def shift(value):\n    return value + 3\n")["status"] == "FAIL"


def test_semantic_grader_accepts_commutative_known_fix_without_hash_match() -> None:
    # This source has different bytes from the retained canonical fix; the oracle
    # evaluates the operation rather than requiring equality to a known patch hash.
    result = grade_bytes(b"def shift(value):\n    return 2 + value\n")
    assert result == {"status": "PASS", "reason": "exact_frozen_shift_contract"}


def test_semantic_grader_rejects_side_effects_and_extra_functions() -> None:
    side_effect = b"def shift(value):\n    print(value)\n    return value + 2\n"
    extra_function = b"def shift(value):\n    return value + 2\ndef helper():\n    return 1\n"
    assert grade_bytes(side_effect)["status"] == "FAIL"
    assert grade_bytes(extra_function)["status"] == "FAIL"


def test_candidate_reader_rejects_symlink(tmp_path) -> None:
    (tmp_path / "real.py").write_text("def shift(value): return value + 2\n")
    os.symlink("real.py", tmp_path / "subject.py")
    source, error = read_candidate(str(tmp_path), "subject.py")
    assert source is None
    assert error == "source_symlink_rejected"
