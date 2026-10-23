from __future__ import annotations

from pathlib import Path

from scripts import calibrate_phase_a_v6_colima_mount_v1 as calibration


def test_fixture_uses_current_public_runner_inventory_contract(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "test_public.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_public_tests(workspace, image, test_hash, memory_mb, **kwargs):
        calls.append(kwargs)
        return {"classification": "assertion_failure", "returncode": 1, "passed": False}

    monkeypatch.setattr(calibration, "run_public_tests", fake_run_public_tests)
    result = calibration._run_fixture(workspace, "def answer(): return 0\n", "test-hash")

    assert calls == [{"expected_workspace_files": {"src/module.py", "test_public.py"}}]
    assert result["result"]["classification"] == "assertion_failure"
