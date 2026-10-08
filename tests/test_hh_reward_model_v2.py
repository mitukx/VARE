from __future__ import annotations

import importlib.util
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from scripts.run_cpu_hh_reward_model_v2 import fit_temperature
from scripts.run_cpu_hh_reward_model import iter_rows
from scripts import audit_cpu_hh_reward_model_v2 as audit_v2

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads((ROOT / "protocols/cpu_hh_reward_model_v2.json").read_text())


def test_temperature_is_positive_bounded_and_train_fit_reduces_nll():
    if torch is None:
        import pytest
        pytest.skip("PyTorch is an optional local-only dependency for HH model-fit checks")
    margins = torch.tensor([1.2, -0.4, 0.8, -0.2, 0.5, 0.1, -0.3, 0.7])
    alpha, diagnostics = fit_temperature(margins, protocol(), torch)
    assert math.isfinite(alpha)
    assert 0 < alpha < 10
    assert diagnostics["oof_nll_after"] < diagnostics["oof_nll_before"]
    assert diagnostics["iterations"] > 0


def test_perfect_oof_ranking_cannot_drive_temperature_above_frozen_cap():
    if torch is None:
        import pytest
        pytest.skip("PyTorch is an optional local-only dependency for HH model-fit checks")
    margins = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    alpha, diagnostics = fit_temperature(margins, protocol(), torch)
    assert 0 < alpha < 10
    assert math.isfinite(diagnostics["oof_nll_after"])


def test_v2_uses_unread_test_ranges_and_excludes_v1_development_hashes():
    spec = protocol()
    selection = spec["dataset"]["selection"]
    prior = selection["prior_development_exclusion"]
    assert selection["development_test_indices"] == [512, 1024]
    assert selection["confirmation_test_indices"] == [1024, 1536]
    assert prior["source_range"] == [0, 512]
    assert len(prior["context_hashes"]) == len(set(prior["context_hashes"])) == 256
    assert spec["dataset"]["max_source_row_index_test"] == 1535
    access = spec["dataset"]["source_hashing_rule"]
    assert "opaque bytes" in access
    assert "no confirmation-range record may be returned" in access


class GuardedLines:
    def __init__(self, lines, maximum_next_calls):
        self.lines = iter(lines)
        self.maximum_next_calls = maximum_next_calls
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls > self.maximum_next_calls:
            raise AssertionError("reader requested a row outside its frozen range")
        return next(self.lines)


class _FakeContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_runner_and_auditor_stop_before_requesting_row_at_exclusive_end():
    lines = ['{"row": 0}\n', '{"row": 1}\n', '{"row": 2}\n']
    for reader, module in ((iter_rows, "scripts.run_cpu_hh_reward_model.gzip"),
                           (audit_v2.read_range, "scripts.audit_cpu_hh_reward_model_v2.gzip")):
        guarded = GuardedLines(lines, maximum_next_calls=2)
        with patch(module + ".open", return_value=_FakeContext(guarded)):
            assert list(reader(Path("unused.jsonl.gz"), 0, 2)) == [(0, {"row": 0}), (1, {"row": 1})]
        assert guarded.calls == 2


def test_protocol_snapshot_comparison_ignores_unicode_escape_style():
    expected = {"label": "受理", "limits": [512, 1024]}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snapshot.json"
        path.write_text(json.dumps(expected, ensure_ascii=True), encoding="utf-8")
        assert audit_v2.json_snapshot_matches(path, expected)
