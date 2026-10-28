from __future__ import annotations

import os
from pathlib import Path

from scripts import run_local_agent as runner
from vare.agent_edit_v2 import MAX_SOURCE_BYTES


def _write(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_check_syntax_reads_only_exact_allowlisted_python(tmp_path: Path) -> None:
    source = _write(tmp_path, "src/good.py", b"answer = 42\n")
    result = runner.check_syntax_source(tmp_path, "src/good.py", {"src/good.py"})
    assert result == {"syntax_valid": True, "path": "src/good.py"}
    assert source.read_bytes() == b"answer = 42\n"

    outside = _write(tmp_path, "outside/secret.py", b"secret_reference = 'must not be read'\n")
    rejected = runner.check_syntax_source(
        tmp_path, "outside/secret.py", {"src/good.py"},
    )
    assert rejected["error"] == "path is not allowed"
    assert "secret_reference" not in str(rejected)
    assert outside.is_file()


def test_check_syntax_rejects_allowlisted_final_symlink_to_outside(tmp_path: Path) -> None:
    outside = _write(tmp_path, "outside/secret.py", b"secret_reference = 'must not leak'\n")
    linked = tmp_path / "src" / "linked.py"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)

    result = runner.check_syntax_source(tmp_path, "src/linked.py", {"src/linked.py"})
    assert result.get("syntax_valid") is not True
    assert "secret_reference" not in str(result)
    assert outside.read_bytes() == b"secret_reference = 'must not leak'\n"


def test_check_syntax_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    outside = _write(tmp_path, "outside/secret.py", b"secret_reference = 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "linked_dir").symlink_to(outside.parent, target_is_directory=True)

    result = runner.check_syntax_source(
        tmp_path, "src/linked_dir/secret.py", {"src/linked_dir/secret.py"},
    )
    assert result.get("syntax_valid") is not True
    assert "secret_reference" not in str(result)


def test_check_syntax_rejects_final_symlink_swap_after_parent_open(
    monkeypatch, tmp_path: Path,
) -> None:
    outside = _write(tmp_path, "outside/secret.py", b"secret_reference = 1\n")
    target = _write(tmp_path, "src/target.py", b"candidate_value = 1\n")
    original_read = runner._read_current
    swapped = False

    def swap_then_read(parent_fd: int, basename: str):
        nonlocal swapped
        if not swapped:
            parked = target.with_suffix(".parked")
            os.replace(target, parked)
            target.symlink_to(outside)
            swapped = True
        return original_read(parent_fd, basename)

    monkeypatch.setattr(runner, "_read_current", swap_then_read)
    result = runner.check_syntax_source(tmp_path, "src/target.py", {"src/target.py"})
    assert swapped
    assert result.get("syntax_valid") is not True
    assert "secret_reference" not in str(result)


def test_check_syntax_enforces_source_byte_limit(tmp_path: Path) -> None:
    _write(tmp_path, "src/oversized.py", b" " * (MAX_SOURCE_BYTES + 1))
    result = runner.check_syntax_source(
        tmp_path, "src/oversized.py", {"src/oversized.py"},
    )
    assert result.get("syntax_valid") is not True
    assert "exceeds byte limit" in str(result)


def test_syntax_error_response_does_not_return_source_line(tmp_path: Path) -> None:
    _write(tmp_path, "src/broken.py", b"secret_assignment = (\n")
    result = runner.check_syntax_source(
        tmp_path, "src/broken.py", {"src/broken.py"},
    )
    assert result["syntax_valid"] is False
    assert result["error"]["message"]
    assert "secret_assignment" not in str(result)


def test_controller_identity_tracks_edit_format_module_drift(monkeypatch, tmp_path: Path) -> None:
    script = _write(tmp_path, "scripts/run_local_agent.py", b"controller\n")
    monkeypatch.setattr(runner, "__file__", str(script))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    names = (
        "src/vare/agent_edit_v2.py",
        "src/vare/agent_edit_format.py",
        "src/vare/agent_task_contract.py",
        "src/vare/source_navigation_v1.py",
    )
    paths = {name: _write(tmp_path, name, f"original:{name}\n".encode()) for name in names}

    before = runner.controller_source_hashes("lease_bound_navigation_v1")
    edit_format_path = paths["src/vare/agent_edit_format.py"]
    edit_format_path.write_bytes(edit_format_path.read_bytes() + b"# drift\n")
    after = runner.controller_source_hashes("lease_bound_navigation_v1")

    assert before["src/vare/agent_edit_format.py"] != after["src/vare/agent_edit_format.py"]
