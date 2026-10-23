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
