from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import vare.agent_runtime_snapshot_v1 as runtime


def _source_tree(root: Path) -> tuple[list[str], dict[str, str]]:
    source = root / "scripts" / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"print('runner')\n")
    return ["scripts/runner.py"], {"example.module": str(source)}


def test_runtime_snapshot_is_atomic_and_auditable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    data = (root / paths[0]).read_bytes()
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()

    captured = runtime.capture_runtime_snapshot(
        repository_root=root,
        output_dir=destination,
        source_paths=paths,
        module_origins=origins,
        expected_hashes={paths[0]: hashlib.sha256(data).hexdigest()},
    )

    audit = runtime.audit_runtime_snapshot(destination)
    assert audit["status"] == "PASS"
    assert captured["manifest_sha256"] == audit["manifest_sha256"]
    assert audit["verified_source_hashes"][paths[0]] == hashlib.sha256(data).hexdigest()
    assert (destination / "source" / paths[0]).read_bytes() == data


def test_runtime_snapshot_rejects_missing_source_without_publication(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()

    with pytest.raises(runtime.RuntimeSnapshotError, match="cannot be resolved"):
        runtime.capture_runtime_snapshot(
            repository_root=root,
            output_dir=destination,
            source_paths=["scripts/missing.py"],
            module_origins={},
        )

    assert not destination.exists()


def test_runtime_snapshot_rejects_module_shadowing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, _origins = _source_tree(root)
    shadow = tmp_path / "shadow.py"
    shadow.write_bytes((root / paths[0]).read_bytes())
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()

    with pytest.raises(runtime.RuntimeSnapshotError, match="shadowed outside repository"):
        runtime.capture_runtime_snapshot(
            repository_root=root,
            output_dir=destination,
            source_paths=paths,
            module_origins={"example.module": str(shadow)},
        )

    assert not destination.exists()


def test_runtime_snapshot_rejects_false_expected_identity(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()

    with pytest.raises(runtime.RuntimeSnapshotError, match="source drift detected"):
        runtime.capture_runtime_snapshot(
            repository_root=root,
            output_dir=destination,
            source_paths=paths,
            module_origins=origins,
            expected_hashes={paths[0]: "0" * 64},
        )

    assert not destination.exists()


def test_runtime_snapshot_detects_tampering_and_missing_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()
    runtime.capture_runtime_snapshot(
        repository_root=root,
        output_dir=destination,
        source_paths=paths,
        module_origins=origins,
    )
    archived = destination / "source" / paths[0]
    archived.unlink()

    with pytest.raises(runtime.RuntimeSnapshotError, match="missing or not a regular file"):
        runtime.audit_runtime_snapshot(destination)


def test_runtime_snapshot_rejects_manifest_digest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()
    runtime.capture_runtime_snapshot(
        repository_root=root,
        output_dir=destination,
        source_paths=paths,
        module_origins=origins,
    )
    (destination / runtime.MANIFEST_DIGEST_NAME).write_text("0" * 64 + "\n")

    with pytest.raises(runtime.RuntimeSnapshotError, match="manifest digest mismatch"):
        runtime.audit_runtime_snapshot(destination)


def test_runtime_snapshot_records_grader_identity_without_copying_grader_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    grader = root / "hidden_grader.py"
    grader.write_bytes(b"expected = 'private'\n")
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()
    grader_sha = hashlib.sha256(grader.read_bytes()).hexdigest()

    result = runtime.capture_runtime_snapshot(
        repository_root=root,
        output_dir=destination,
        source_paths=paths,
        module_origins=origins,
        grader_entrypoints=[grader],
    )

    assert result["grader_entrypoints"] == [{"path": "hidden_grader.py", "sha256": grader_sha}]
    assert not (destination / "source" / "hidden_grader.py").exists()
    assert runtime.audit_runtime_snapshot(destination)["status"] == "PASS"


def test_runtime_snapshot_cleans_partial_staging_on_write_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths, origins = _source_tree(root)
    destination = tmp_path / "attempt" / "runtime_snapshot"
    destination.parent.mkdir()
    original = runtime._write_exclusive

    def fail_manifest(path: Path, data: bytes) -> None:
        if path.name == runtime.MANIFEST_NAME:
            raise OSError("injected manifest write failure")
        original(path, data)

    monkeypatch.setattr(runtime, "_write_exclusive", fail_manifest)
    with pytest.raises(OSError, match="injected manifest write failure"):
        runtime.capture_runtime_snapshot(
            repository_root=root,
            output_dir=destination,
            source_paths=paths,
            module_origins=origins,
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".runtime-snapshot-*"))
