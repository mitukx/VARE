from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.grade_hf_coding_dev_v1 import _verify_runtime_lock
from scripts.grade_transformers_phase_a_v1 import _scan_candidate
from scripts.run_local_agent import (
    classify_public_test_error,
    classify_public_test_exit,
    public_test_infrastructure_error,
)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _candidate(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "candidate"
    source = root / "src" / "package"
    source.mkdir(parents=True)
    target = source / "target.py"
    protected = source / "dependency.py"
    target.write_bytes(b"def answer():\n    return 1\n")
    protected.write_bytes(b"VALUE = 7\n")
    for path in (target, protected):
        path.chmod(0o644)
    for path in (root / "src", source):
        path.chmod(0o755)
    manifest: dict[str, object] = {
        "files": {
            "src/package/target.py": {"sha256": _digest(target.read_bytes()), "mode": 0o644},
            "src/package/dependency.py": {
                "sha256": _digest(protected.read_bytes()), "mode": 0o644,
            },
        },
        "directories": {"src": 0o755, "src/package": 0o755},
    }
    return root, manifest


def test_candidate_inventory_accepts_only_the_declared_target_change(tmp_path: Path) -> None:
    root, manifest = _candidate(tmp_path)
    target = root / "src/package/target.py"
    target.write_bytes(b"def answer():\n    return 2\n")

    files, file_count, directory_count = _scan_candidate(
        root, manifest, "src/package/target.py",
    )

    assert files["src/package/target.py"] == target.read_bytes()
    assert file_count == 2
    assert directory_count == 2


@pytest.mark.parametrize(
    "attack",
    [
        "sitecustomize",
        "altered-import",
        "symlink",
        "hidden-git",
        "forged-evidence",
        "missing-file",
        "unexpected-executable",
        "target-mode-change",
        "directory-symlink",
    ],
)
def test_candidate_inventory_rejects_untrusted_workspace_changes(
    tmp_path: Path, attack: str,
) -> None:
    root, manifest = _candidate(tmp_path)
    if attack == "sitecustomize":
        (root / "sitecustomize.py").write_text("import os\n", encoding="utf-8")
    elif attack == "altered-import":
        (root / "src/package/dependency.py").write_text("VALUE = 0\n", encoding="utf-8")
    elif attack == "symlink":
        (root / "src/package/extra.py").symlink_to("dependency.py")
    elif attack == "hidden-git":
        (root / ".git").mkdir()
    elif attack == "forged-evidence":
        (root / "grade-result.json").write_text('{"passed":true}\n', encoding="utf-8")
    elif attack == "missing-file":
        (root / "src/package/dependency.py").unlink()
    elif attack == "unexpected-executable":
        executable = root / "src/package/extra.py"
        executable.write_text("print('untrusted startup')\n", encoding="utf-8")
        executable.chmod(0o755)
    elif attack == "target-mode-change":
        (root / "src/package/target.py").chmod(0o755)
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (root / "src/package").rename(root / "src/package-pinned")
        (root / "src/package").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        _scan_candidate(root, manifest, "src/package/target.py")


def test_public_test_exit_codes_keep_infrastructure_failures_out_of_reward() -> None:
    assert classify_public_test_exit(0) == "passed"
    assert classify_public_test_exit(1) == "test_failure"
    for returncode in (125, 126, 127):
        result = {
            "passed": False,
            "classification": classify_public_test_exit(returncode),
        }
        assert public_test_infrastructure_error(result)
    assert not public_test_infrastructure_error(
        {"passed": False, "classification": classify_public_test_exit(1)},
    )


def test_public_test_exception_classification_is_fail_closed() -> None:
    assert classify_public_test_error("could not confirm Docker container cleanup") == (
        "evaluator_unavailable"
    )
    assert classify_public_test_error("public test file is missing or changed") == (
        "runner_integrity_error"
    )


def test_candidate_screen_runtime_build_inputs_match_the_frozen_lock() -> None:
    protocol = json.loads(
        Path("benchmarks/hf_coding_dev_v1/protocol-v1.json").read_text(encoding="utf-8"),
    )
    runtime = protocol["runtime"]

    _verify_runtime_lock(runtime)

    altered = dict(runtime)
    altered["requirements_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime input differs from lock"):
        _verify_runtime_lock(altered)
