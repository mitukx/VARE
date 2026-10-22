from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vare.agent_structured_edit import (
    StructuredEditError,
    StructuredEditController,
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


def test_controller_returns_short_region_and_opaque_revision_handles(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    read = controller.read_region("src/module.py", start_line=1, end_line=2)
    assert read["content"] == "def value():\n    return 1\n"
    assert "workspace_revision" not in read
    assert "source_preimage" not in read
    assert read["workspace_revision_handle"].startswith("rev-")
    assert read["source_preimage_handle"].startswith("src-")
    assert len(read["workspace_revision_handle"]) < 16
    assert len(read["source_preimage_handle"]) < 16

    result = controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=read["workspace_revision_handle"],
        source_preimage_handle=read["source_preimage_handle"],
        old_text="return 1",
        new_text="return 2",
    )
    assert result["applied"] is True
    assert "expected_hash" not in result
    assert "expected_revision" not in result


def test_controller_rejects_stale_or_cross_file_handles(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    read = controller.read_region("src/module.py")
    (root / "src/module.py").write_text("def value():\n    return 9\n", encoding="utf-8")
    with pytest.raises(StructuredEditError, match="revision handle is stale"):
        controller.apply_edit(
            action="replace_text",
            path="src/module.py",
            workspace_revision_handle=read["workspace_revision_handle"],
            source_preimage_handle=read["source_preimage_handle"],
            old_text="return 1",
            new_text="return 2",
        )
    (root / "src/other.py").write_text("value = 1\n", encoding="utf-8")
    allowed.add("src/other.py")
    cross_controller = StructuredEditController(root, allowed)
    module = cross_controller.read_region("src/module.py")
    other = cross_controller.read_region("src/other.py")
    with pytest.raises(StructuredEditError, match="same source snapshot"):
        cross_controller.apply_edit(
            action="replace_text",
            path="src/other.py",
            workspace_revision_handle=module["workspace_revision_handle"],
            source_preimage_handle=other["source_preimage_handle"],
            old_text="return 9",
            new_text="return 10",
        )


def test_controller_binds_omitted_handles_to_the_latest_read_for_that_path(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    controller.read_region("src/module.py", start_line=1, end_line=2)
    # A second source read refreshes the host-managed handle for this path.
    controller.read_region("src/module.py", start_line=1, end_line=2)

    result = controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        old_text="return 1",
        new_text="return 2",
    )

    assert result["applied"] is True
    assert "return 2" in (root / "src/module.py").read_text(encoding="utf-8")


def test_controller_rejects_partial_or_unbound_implicit_handles(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    with pytest.raises(StructuredEditError, match="before editing"):
        controller.apply_edit(
            action="replace_text",
            path="src/module.py",
            old_text="return 1",
            new_text="return 2",
        )
    controller.read_region("src/module.py")
    with pytest.raises(StructuredEditError, match="both edit handles"):
        controller.apply_edit(
            action="replace_text",
            path="src/module.py",
            workspace_revision_handle="rev-0001",
            old_text="return 1",
            new_text="return 2",
        )


def test_implicit_handles_still_reject_stale_workspace_revisions(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    other = root / "src/other.py"
    other.write_text("value = 1\n", encoding="utf-8")
    allowed.add("src/other.py")
    controller = StructuredEditController(root, allowed)
    controller.read_region("src/module.py")
    other.write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(StructuredEditError, match="revision handle is stale"):
        controller.apply_edit(
            action="replace_text",
            path="src/module.py",
            old_text="return 1",
            new_text="return 2",
        )


def test_controller_bounds_region_size_and_source_file_allowlist(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed, max_read_lines=1, max_read_chars=256)
    read = controller.read_region("src/module.py", start_line=1, end_line=3)
    assert read["content"] == "def value():\n"
    assert read["end_line"] == 1
    assert read["next_line"] == 2
    assert read["truncated"] is True
    with pytest.raises(StructuredEditError, match="allowlisted"):
        controller.read_region("outside.py")
    with pytest.raises(ValueError, match="controller bounds"):
        StructuredEditController(root, allowed, max_read_lines=121)


def test_controller_can_read_but_never_edit_public_test_file(tmp_path: Path) -> None:
    root, editable = _workspace(tmp_path)
    public_test = root / "test_public.py"
    public_test.write_text("assert value() == 2\n", encoding="utf-8")
    controller = StructuredEditController(
        root, editable, readable_files=editable | {"test_public.py"},
    )
    read = controller.read_region("test_public.py")
    assert read["content"] == "assert value() == 2\n"
    assert read["read_only"] is True
    assert "workspace_revision" not in read
    with pytest.raises(StructuredEditError, match="read the current source region"):
        controller.apply_edit(
            action="replace_text",
            path="test_public.py",
            workspace_revision_handle="",
            source_preimage_handle="",
            old_text="2",
            new_text="1",
        )


def test_controller_bounds_generated_edit_text(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    read = controller.read_region("src/module.py")
    with pytest.raises(StructuredEditError, match="per-action character budget"):
        controller.apply_edit(
            action="replace_text",
            path="src/module.py",
            workspace_revision_handle=read["workspace_revision_handle"],
            source_preimage_handle=read["source_preimage_handle"],
            old_text="return 1",
            new_text="2" * 8_001,
        )


def test_controller_detects_repeated_identical_noop_edits(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    for attempt in range(2):
        read = controller.read_region("src/module.py")
        with pytest.raises(StructuredEditError) as error:
            controller.apply_edit(
                action="replace_text",
                path="src/module.py",
                workspace_revision_handle=read["workspace_revision_handle"],
                source_preimage_handle=read["source_preimage_handle"],
                old_text="return 1",
                new_text="return 1",
            )
        if attempt == 1:
            assert "repeated no-op" in str(error.value)


def test_controller_repeated_public_failure_prompts_different_repair_and_bounds_output(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed, max_feedback_chars=128)
    read = controller.read_region("src/module.py")
    controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=read["workspace_revision_handle"],
        source_preimage_handle=read["source_preimage_handle"],
        old_text="return 1",
        new_text="return 2",
    )
    first = controller.record_public_test_result({"passed": False, "stderr": "assertion failed"})
    second = controller.record_public_test_result({"passed": False, "stderr": "assertion failed"})
    assert first["repeated_unchanged_failure"] is False
    assert second["repeated_unchanged_failure"] is True
    assert second["next_action"] == "inspect the failing assertion and make a different targeted edit"
    assert len(second["feedback"]) <= 128
    assert len(second["failure_signature"]) == 64


def test_controller_does_not_turn_container_launch_failure_into_test_feedback(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    read = controller.read_region("src/module.py")
    controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=read["workspace_revision_handle"],
        source_preimage_handle=read["source_preimage_handle"],
        old_text="return 1",
        new_text="return 2",
    )

    feedback = controller.record_public_test_result({
        "passed": None,
        "classification": "container_launch_failure",
        "gradeable": False,
        "error": "bind source path does not exist: /private/var/folders/secret",
    })

    assert feedback["classification"] == "container_launch_failure"
    assert "no test assertion ran" in feedback["feedback"]
    assert "secret" not in feedback["feedback"]
    assert feedback["next_action"].startswith("stop this rollout")
    assert feedback["passed"] is None
    assert "secret" not in feedback["feedback"]
    with pytest.raises(StructuredEditError, match="until public tests pass"):
        controller.finalize_for_independent_grading()


def test_finalization_requires_passed_public_tests_and_inspected_current_diff(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    read = controller.read_region("src/module.py")
    controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=read["workspace_revision_handle"],
        source_preimage_handle=read["source_preimage_handle"],
        old_text="return 1",
        new_text="return 2",
    )
    with pytest.raises(StructuredEditError, match="until public tests pass"):
        controller.finalize_for_independent_grading()
    controller.record_public_test_result({"passed": True})
    with pytest.raises(StructuredEditError, match="inspect the current diff"):
        controller.finalize_for_independent_grading()
    controller.inspect_diff()
    final = controller.finalize_for_independent_grading()
    assert final["candidate_ready_for_independent_grading"] is True
    assert final["verified_success"] is False


def test_new_edit_invalidates_prior_public_pass_and_diff_inspection(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    first = controller.read_region("src/module.py")
    controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=first["workspace_revision_handle"],
        source_preimage_handle=first["source_preimage_handle"],
        old_text="return 1",
        new_text="return 2",
    )
    controller.record_public_test_result({"passed": True})
    controller.inspect_diff()

    current = controller.read_region("src/module.py")
    controller.apply_edit(
        action="replace_text",
        path="src/module.py",
        workspace_revision_handle=current["workspace_revision_handle"],
        source_preimage_handle=current["source_preimage_handle"],
        old_text="return 2",
        new_text="return 3",
    )
    with pytest.raises(StructuredEditError, match="until public tests pass"):
        controller.finalize_for_independent_grading()


def test_public_test_feedback_requires_trusted_boolean_result(tmp_path: Path) -> None:
    root, allowed = _workspace(tmp_path)
    controller = StructuredEditController(root, allowed)
    with pytest.raises(StructuredEditError, match="trusted test runner"):
        controller.record_public_test_result({"passed": "true", "stderr": ""})

