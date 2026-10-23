from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

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


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r", b""])
@pytest.mark.parametrize("embedded", [b"", b"\r", "\u2028".encode()])
def test_generated_diffs_replay_exact_source_bytes(
    tmp_path: Path, ending: bytes, embedded: bytes,
) -> None:
    root, allowed = _workspace(tmp_path)
    path = root / "src/module.py"
    original = b"# " + embedded + (ending or b"\n") + b"def value():" + (ending or b"\n") + b"    return 1" + ending
    path.write_bytes(original)
    before = {"src/module.py": original}
    assert unified_diff(root, before, allowed) == ""
    result = apply_structured_edit(root, allowed, **_edit(root, allowed))
    expected = original.replace(b"return 1", b"return 2")
    assert path.read_bytes() == expected
    snapshot_diff = unified_diff(root, before, allowed)
    assert snapshot_diff == result["diff"]
    assert result["diff_sha256"] == hashlib.sha256(snapshot_diff.encode()).hexdigest()
    for patch in (result["diff"], snapshot_diff):
        path.write_bytes(original)
        subprocess.run(
            ["git", "apply", "--check", "-"], cwd=root,
            input=patch.encode(), capture_output=True, check=True, timeout=10,
        )
        subprocess.run(
            ["git", "apply", "-"], cwd=root,
            input=patch.encode(), capture_output=True, check=True, timeout=10,
        )
        assert path.read_bytes() == expected


def test_symlink_in_workspace_ancestor_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    workspace = actual / "workspace"
    workspace.mkdir()
    root, allowed = _workspace(workspace)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    aliased_root = alias / "workspace"
    assert not aliased_root.is_symlink()
    original = (root / "src/module.py").read_bytes()
    with pytest.raises(StructuredEditError, match="symlinked workspace"):
        workspace_revision(aliased_root, allowed)
    with pytest.raises(StructuredEditError, match="symlinked workspace"):
        unified_diff(aliased_root, {"src/module.py": original}, allowed)
    with pytest.raises(StructuredEditError, match="symlinked workspace"):
        apply_structured_edit(
            aliased_root, allowed, action="replace_text", path="src/module.py",
            expected_hash=hashlib.sha256(original).hexdigest(),
            expected_revision=workspace_revision(root, allowed),
            old_text="return 1", new_text="return 2",
        )
    assert (root / "src/module.py").read_bytes() == original


@pytest.mark.parametrize("character", ["\t", "\n", "\r", "\x1b", "\x7f"])
def test_control_characters_in_allowlisted_paths_are_rejected_before_edit(
    tmp_path: Path, character: str,
) -> None:
    name = f"source{character}.py"
    target = tmp_path / name
    original = b"return 1\n"
    target.write_bytes(original)
    with pytest.raises(StructuredEditError, match="allowlisted"):
        apply_structured_edit(
            tmp_path, {name}, action="replace_text", path=name,
            expected_hash=hashlib.sha256(original).hexdigest(),
            expected_revision="0" * 64, old_text="return 1", new_text="return 2",
        )
    assert target.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", ["./src/module.py", "src/./module.py", "src//module.py", "src/module.py/"])
def test_noncanonical_allowlist_entries_are_rejected(tmp_path: Path, name: str) -> None:
    root, _ = _workspace(tmp_path)
    target = root / "src/module.py"
    before = target.read_bytes()
    with pytest.raises(StructuredEditError, match="allowlisted"):
        apply_structured_edit(
            root, {name}, action="replace_text", path=name,
            expected_hash=hashlib.sha256(before).hexdigest(),
            expected_revision="0" * 64, old_text="return 1", new_text="return 2",
        )
    assert target.read_bytes() == before


@pytest.mark.parametrize("name", ["space name.py", "unicode-\u00fc.py", 'quote"name.py', "colon:name.py"])
def test_canonical_unusual_filenames_generate_applicable_diffs(tmp_path: Path, name: str) -> None:
    original = b"return 1\n"
    target = tmp_path / name
    target.write_bytes(original)
    result = apply_structured_edit(
        tmp_path, {name}, action="replace_text", path=name,
        expected_hash=hashlib.sha256(original).hexdigest(),
        expected_revision=workspace_revision(tmp_path, {name}),
        old_text="return 1", new_text="return 2",
    )
    target.write_bytes(original)
    subprocess.run(
        ["git", "apply", "-"], cwd=tmp_path,
        input=result["diff"].encode(), capture_output=True, check=True, timeout=10,
    )
    assert target.read_bytes() == b"return 2\n"


@pytest.mark.parametrize("action", ["replace_text", "insert_text"])
def test_overlapping_text_matches_are_rejected_without_mutation(tmp_path: Path, action: str) -> None:
    target = tmp_path / "source.py"
    target.write_bytes(b"aaa")
    allowed = {"source.py"}
    with pytest.raises(StructuredEditError, match="match exactly once"):
        apply_structured_edit(
            tmp_path, allowed, action=action, path="source.py",
            expected_hash=hashlib.sha256(b"aaa").hexdigest(),
            expected_revision=workspace_revision(tmp_path, allowed),
            old_text="aa", anchor="aa", new_text="X",
        )
    assert target.read_bytes() == b"aaa"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("name", [".git/config", "dir/.git/config", "dir/.GiT/config"])
def test_git_metadata_components_are_rejected_before_mutation(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    target.parent.mkdir(parents=True)
    original = b"value=1\n"
    target.write_bytes(original)
    with pytest.raises(StructuredEditError, match="allowlisted"):
        apply_structured_edit(
            tmp_path, {name}, action="replace_text", path=name,
            expected_hash=hashlib.sha256(original).hexdigest(),
            expected_revision="0" * 64, old_text="value=1", new_text="value=2",
        )
    assert target.read_bytes() == original
