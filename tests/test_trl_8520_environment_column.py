"""Independent in-memory controls for the restricted source replay."""
import pytest

from scripts import reproduce_trl_8520_environment_column as R


def fixture(columns, *, guard=True):
    body = f"self._signature_columns = {columns!r}"
    if guard:
        body = "if self._signature_columns is None:\n            " + body
    source = "class GRPOTrainer:\n    def _set_signature_columns_if_needed(self):\n        " + body
    return R.method_from_source(source, "GRPOTrainer", "_set_signature_columns_if_needed")


def test_environment_projection_contract_and_bad_controls():
    baseline = fixture(["prompt", "image", "images"])
    fixed = fixture(["prompt", "image", "images", "environment"])
    assert not R.exercise(R.checked_method(baseline))["environment_only"]
    assert all(R.exercise(R.checked_method(fixed)).values())
    for name in ("wrong_plural", "drop_image", "overwrite_existing"):
        assert not all(R.exercise(R.checked_method(
            R.mutant(fixed, name), allow_unconditional=name == "overwrite_existing",
        )).values())


def test_arbitrary_assignment_expression_is_refused_before_execution():
    source = """class GRPOTrainer:
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = unknown()
"""
    method = R.method_from_source(source, "GRPOTrainer", "_set_signature_columns_if_needed")
    with pytest.raises(ValueError, match="literal"):
        R.checked_method(method)
