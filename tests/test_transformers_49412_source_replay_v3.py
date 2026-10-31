from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reproduce_transformers_49412_source_replay_v3 as replay


def _registration_module(*, fail_first_call: bool) -> SimpleNamespace:
    registry: dict[str, list[replay.Transform]] = {}
    calls = 0

    def register(_model, _module, _fusion_name, spec) -> None:
        nonlocal calls
        calls += 1
        if fail_first_call:
            raise ValueError("unrelated first-registration initialization failure")
        if calls == 1:
            registry[replay.FakeConfig.model_type] = spec.make_transforms(
                replay.FakeConfig
            )
            return
        raise ValueError(replay.EXPECTED_BASELINE_CONFLICT)

    return SimpleNamespace(
        _probe_registry=registry,
        _FUSION_DISCOVERY_CACHE={},
        _REGISTERED_FUSION_SOURCE_PATTERNS={},
        _register_module_fusion=register,
        _probe_executed_source_sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("fail_first_call", "expected_exit", "expected_result"),
    [
        (False, 0, "BASELINE_FAIL_REPRODUCED"),
        (True, 1, "FAIL"),
    ],
)
def test_v3_baseline_classifier_requires_the_second_registration_conflict(
    fail_first_call: bool,
    expected_exit: int,
    expected_result: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "fusion_mapping.py"
    source.write_text("# isolated classifier fixture\n", encoding="utf-8")
    module = _registration_module(fail_first_call=fail_first_call)
    monkeypatch.setattr(
        replay,
        "load_module",
        lambda *_args: (module, {}, {}),
    )
    monkeypatch.setattr(replay, "unload_module", lambda *_args: None)

    exit_code = replay.run_case("baseline", source)
    record = json.loads(capsys.readouterr().out)
    assert exit_code == expected_exit
    assert record["result"] == expected_result
    if fail_first_call:
        assert record["properties_passed"] == []
        assert record["property_failures"] == [{
            "property": "baseline_conflict_occurs_only_on_second_registration",
            "error": "ValueError: unrelated first-registration initialization failure",
        }]
    else:
        assert record["properties_passed"] == [
            "baseline_conflict_occurs_only_on_second_registration"
        ]
        assert record["property_failures"] == []
