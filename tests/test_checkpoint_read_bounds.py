"""Fresh IO faults, independent of the retired model evaluation cohorts."""
from __future__ import annotations

import os
import subprocess
import sys
import tracemalloc

import pytest

from vare.checkpoint_artifacts import CheckpointIntegrityError, TrustedCheckpointStore


def registered(tmp_path, raw=b"checkpoint-v1"):
    store = TrustedCheckpointStore(tmp_path / "store", max_bytes=128 * 1024**2)
    source = tmp_path / "source"
    source.write_bytes(raw)
    digest = store.register_file("p0", source)
    return store, source, store.objects / digest


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO fault")
@pytest.mark.parametrize("operation", ["register", "digest", "read"])
def test_fifo_is_rejected_without_waiting_for_writer(tmp_path, operation):
    store, source, obj = registered(tmp_path)
    target = source if operation == "register" else obj
    target.unlink()
    os.mkfifo(target)
    code = """
import sys
from vare.checkpoint_artifacts import TrustedCheckpointStore, CheckpointIntegrityError
store = TrustedCheckpointStore(sys.argv[1])
print('ready', flush=True)
try:
    if sys.argv[2] == 'register':
        store.register_file('p1', sys.argv[3])
    elif sys.argv[2] == 'digest':
        store.digest('p0')
    else:
        store.read_verified('p0')
except CheckpointIntegrityError:
    print('rejected', flush=True)
else:
    raise AssertionError('FIFO was accepted')
"""
    # External timeout bounds the broken baseline too; no FIFO writer exists.
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(store.root), operation, str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=tmp_path,
    )
    try:
        stdout, stderr = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail(f"checkpoint {operation} blocked on FIFO: {stdout} {stderr}")
    assert proc.returncode == 0, stderr
    assert stdout.splitlines() == ["ready", "rejected"]


def test_registered_size_caps_in_memory_read_after_replacement(tmp_path):
    store, _, obj = registered(tmp_path)
    # An actual 32 MiB regular file replaces the small registered object.
    with obj.open("wb") as out:
        block = b"x" * 1024**2
        for _ in range(32):
            out.write(block)
    tracemalloc.start()
    try:
        with pytest.raises(CheckpointIntegrityError):
            store.read_verified("p0", max_read_bytes=64)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024, f"small checkpoint read allocated {peak} bytes"


@pytest.mark.parametrize("operation", ["digest", "read"])
def test_growth_after_descriptor_stat_is_bounded(tmp_path, monkeypatch, operation):
    store, _, obj = registered(tmp_path)
    real_fstat = os.fstat
    payload = b"x" * (8 * 1024**2)
    fired = False

    def grow_after_stat(fd):
        nonlocal fired
        info = real_fstat(fd)
        if not fired and info.st_ino == obj.stat().st_ino:
            fired = True
            with obj.open("ab") as out:
                out.write(payload)
        return info

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    tracemalloc.start()
    try:
        with pytest.raises(CheckpointIntegrityError):
            store.digest("p0") if operation == "digest" else store.read_verified("p0", max_read_bytes=64)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert fired, "fault did not touch the checkpoint descriptor"
    assert peak < 512 * 1024, f"growing checkpoint read allocated {peak} bytes"


@pytest.mark.parametrize("raw", [b"", b"z", b"a" * 1024**2], ids=["empty", "one-byte", "one-mib"])
def test_exact_bound_and_empty_checkpoint_controls(tmp_path, raw):
    store, _, obj = registered(tmp_path, raw)
    loaded, digest = store.read_verified("p0", max_read_bytes=max(1, len(raw)))
    assert loaded == raw
    assert store.digest("p0") == digest == obj.name


@pytest.mark.parametrize("fault", ["truncate", "same_size", "symlink", "missing"])
def test_existing_corruption_checks_remain_fail_closed(tmp_path, fault):
    store, source, obj = registered(tmp_path)
    if fault == "truncate":
        obj.write_bytes(b"")
    elif fault == "same_size":
        obj.write_bytes(b"x" * obj.stat().st_size)
    else:
        obj.unlink()
        if fault == "symlink":
            obj.symlink_to(source)
    with pytest.raises(CheckpointIntegrityError):
        store.read_verified("p0")
    with pytest.raises(CheckpointIntegrityError):
        store.digest("p0")
