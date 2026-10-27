import hashlib
import json
import tarfile

from scripts.verify_coding_agent_evidence_manifest_v1 import verify


def _entry(path, content):
    return {"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_verifies_sorted_entry_manifest_schema(tmp_path):
    payload = b"retained raw evidence\n"
    (tmp_path / "agent.json").write_bytes(payload)
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "files": [_entry("agent.json", payload)],
    }))

    assert verify(tmp_path) == (1, [])


def test_preserves_compatibility_with_path_map_manifest_schema(tmp_path):
    payload = b"older retained evidence\n"
    (tmp_path / "agent.json").write_bytes(payload)
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({
        "files": {"agent.json": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
    }))

    assert verify(tmp_path) == (1, [])


def test_rejects_duplicate_paths_in_sorted_entry_manifest(tmp_path):
    payload = b"evidence\n"
    entry = _entry("agent.json", payload)
    (tmp_path / "agent.json").write_bytes(payload)
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({"files": [entry, entry]}))

    verified, failures = verify(tmp_path)

    assert verified == 1
    assert failures == ["duplicate manifest path: agent.json"]


def test_rejects_missing_and_empty_file_inventories(tmp_path):
    for manifest in ({}, {"files": {}}, {"files": []}):
        (tmp_path / "evidence-manifest.json").write_text(json.dumps(manifest))

        verified, failures = verify(tmp_path)

        assert verified == 0
        assert failures


def test_rejects_duplicate_json_keys_in_path_map_manifest(tmp_path):
    payload = b"evidence\n"
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / "agent.json").write_bytes(payload)
    (tmp_path / "evidence-manifest.json").write_text(
        '{"files":{"agent.json":{"bytes":9,"sha256":"'
        + digest
        + '"},"agent.json":{"bytes":9,"sha256":"'
        + digest
        + '"}}}'
    )

    verified, failures = verify(tmp_path)

    assert verified == 0
    assert failures == ["invalid evidence manifest: duplicate JSON object key: agent.json"]


def test_rejects_invalid_manifest_root_and_path_map_values(tmp_path):
    manifest_path = tmp_path / "evidence-manifest.json"
    for contents, expected_failure in (
        ('null', "manifest must be a JSON object"),
        ('{"files":{"agent.json":[]}}', "files['agent.json'] must be an object"),
    ):
        manifest_path.write_text(contents)

        verified, failures = verify(tmp_path)

        assert verified == 0
        assert failures == [expected_failure]


def test_verifies_workspace_files_from_archive_when_git_tree_is_not_portable(tmp_path):
    workspace = tmp_path / "greedy" / "workspace"
    source = workspace / ".git" / "config"
    source.parent.mkdir(parents=True)
    payload = b"[core]\nrepositoryformatversion = 0\n"
    source.write_bytes(payload)
    archive_path = tmp_path / "greedy" / "workspace.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(source, arcname=".git/config")
    source.unlink()
    source.parent.rmdir()
    workspace.rmdir()
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({
        "files": [_entry("greedy/workspace/.git/config", payload)],
    }))

    assert verify(tmp_path) == (1, [])


def test_rejects_symlinked_workspace_file_even_when_archive_has_original(tmp_path):
    workspace = tmp_path / "greedy" / "workspace"
    workspace.mkdir(parents=True)
    retained = b"verified archive bytes\n"
    altered = b"unverified live bytes\n"
    (tmp_path / "greedy" / "workspace.tar").parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tmp_path / "greedy" / "workspace.tar", "w") as archive:
        import io
        info = tarfile.TarInfo("agent.py")
        info.size = len(retained)
        archive.addfile(info, io.BytesIO(retained))
    outside = tmp_path / "altered-agent.py"
    outside.write_bytes(altered)
    (workspace / "agent.py").symlink_to(outside)
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({
        "files": [_entry("greedy/workspace/agent.py", retained)],
    }))

    verified, failures = verify(tmp_path)

    assert verified == 0
    assert failures == ["greedy/workspace/agent.py"]


def test_rejects_symlinked_workspace_ancestor_even_when_archive_has_original(tmp_path):
    greedy = tmp_path / "greedy"
    greedy.mkdir()
    live_workspace = tmp_path / "altered-workspace"
    live_workspace.mkdir()
    (live_workspace / "agent.py").write_bytes(b"unverified live bytes\n")
    (greedy / "workspace").symlink_to(live_workspace, target_is_directory=True)
    retained = b"verified archive bytes\n"
    with tarfile.open(greedy / "workspace.tar", "w") as archive:
        import io
        info = tarfile.TarInfo("agent.py")
        info.size = len(retained)
        archive.addfile(info, io.BytesIO(retained))
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({
        "files": [_entry("greedy/workspace/agent.py", retained)],
    }))

    verified, failures = verify(tmp_path)

    assert verified == 0
    assert failures == ["greedy/workspace/agent.py"]
