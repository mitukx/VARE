from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vare.agent_structured_edit import (
    StructuredEditError,
    apply_structured_edit,
    unified_diff,
    workspace_revision,
)


def _workspace(root: Path) -> tuple[Path, set[str]]:
    (root / "src").mkdir()
    (root / "src" / "module.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    allowed = {"src/module.py"}
    return root, allowed


def _edit(root: Path, allowed: set[str], **overrides: str) -> dict[str, str]:
    path = "src/module.py"
    source = (root / path).read_bytes()
    args = {
        "action": "replace_text",
        "path": path,
        "expected_hash": hashlib.sha256(source).hexdigest(),
        "expected_revision": workspace_revision(root, allowed),
        "old_text": "return 1",
        "new_text": "return 2",
    }
    args.update(overrides)
    return args


def test_replace_emits_vare_generated_diff_and_source_hashes(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    before = {"src/module.py": (root / "src/module.py").read_bytes()}
    result = apply_structured_edit(root, allowed, **_edit(root, allowed))
    assert result["applied"] is True
    assert result["source_sha256_before"] == hashlib.sha256(before["src/module.py"]).hexdigest()
    assert result["source_sha256_after"] == hashlib.sha256((root / "src/module.py").read_bytes()).hexdigest()
    assert "-    return 1" in result["diff"]
    assert "-    return 1" in unified_diff(root, before, allowed)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"expected_hash": "0" * 64}, "preimage hash is stale"),
        ({"expected_revision": "0" * 64}, "workspace revision is stale"),
        ({"old_text": "\n"}, "match exactly once"),
        ({"old_text": "return 1", "new_text": "return 1"}, "would not change"),
    ],
)
def test_invalid_or_stale_edit_preserves_workspace(
    tmp_path: Path, overrides: dict[str, str], error: str,
) -> None:
    root, allowed = _workspace(tmp_path)
    path = root / "src/module.py"
    original = path.read_bytes()
    with pytest.raises(StructuredEditError, match=error):
        apply_structured_edit(root, allowed, **_edit(root, allowed, **overrides))
    assert path.read_bytes() == original
    assert list(path.parent.glob(".module.py.*")) == []


def test_insert_requires_one_anchor_and_changes_only_allowlisted_file(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    args = _edit(
        root, allowed,
        action="insert_text", anchor="def value():\n", old_text="", new_text="    # reviewed\n",
    )
    result = apply_structured_edit(root, allowed, **args)
    assert result["applied"] is True
    assert (root / "src/module.py").read_text(encoding="utf-8") == "def value():\n    # reviewed\n    return 1\n"


@pytest.mark.parametrize("raw_path", ["../module.py", "src/../module.py", "/tmp/secret", "src/other.py"])
def test_path_traversal_and_nonallowlisted_edits_are_rejected(tmp_path: Path, raw_path: str) -> None:
    root, allowed = _workspace(tmp_path)
    with pytest.raises(StructuredEditError, match="allowlisted"):
        apply_structured_edit(root, allowed, **_edit(root, allowed, path=raw_path))
    assert (root / "src/module.py").read_text(encoding="utf-8").endswith("return 1\n")


def test_symlink_target_is_rejected(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    original = root / "src/module.py"
    outside = root / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(StructuredEditError, match="symlink"):
        apply_structured_edit(root, allowed, **_edit(root, allowed))
    assert outside.read_text(encoding="utf-8") == "secret\n"




def test_symlinked_workspace_root_rejected_before_source_read(tmp_path: Path) -> None:
    original_root = tmp_path / "actual"
    original_root.mkdir()
    root, allowed = _workspace(original_root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(StructuredEditError, match="symlinked workspace roots"):
        workspace_revision(alias, allowed)
    with pytest.raises(StructuredEditError, match="symlinked workspace roots"):
        apply_structured_edit(
            alias, allowed, action="replace_text", path="src/module.py",
            expected_hash="0" * 64, expected_revision="0" * 64,
            old_text="return 1", new_text="return 2",
        )
    assert (root / "src/module.py").read_text(encoding="utf-8").endswith("return 1\n")


def test_hardlinked_allowlisted_file_cannot_expose_external_content(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "candidate"
    real_root.mkdir()
    root, allowed = _workspace(real_root)
    external = tmp_path / "private.py"
    external.write_text("private_value = 42\n", encoding="utf-8")
    target = root / "src/module.py"
    target.unlink()
    target.hardlink_to(external)

    with pytest.raises(StructuredEditError, match="hard-linked"):
        workspace_revision(root, allowed)
    with pytest.raises(StructuredEditError, match="hard-linked"):
        apply_structured_edit(
            root, allowed, action="replace_text", path="src/module.py",
            expected_hash="0" * 64, expected_revision="0" * 64,
            old_text="private_value", new_text="changed",
        )
    assert external.read_text(encoding="utf-8") == "private_value = 42\n"
