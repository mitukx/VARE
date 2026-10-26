#!/usr/bin/env python3
"""Verify a coding-agent run manifest, including workspace files stored in tar archives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import tarfile


def digest_stream(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def is_safe_regular_file(path: Path, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return False
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def workspace_archive_entry(root: Path, relative: str) -> tuple[Path, str] | None:
    parts = Path(relative).parts
    if len(parts) >= 3 and parts[0] in {'greedy', 'sampled'} and parts[1] == 'workspace':
        return root / parts[0] / 'workspace.tar', Path(*parts[2:]).as_posix()
    if len(parts) < 4 or parts[0] not in {'greedy', 'sampled'}:
        return None
    if parts[1:3] != ('candidate-work-root', 'workspace'):
        return None
    return root / parts[0] / 'workspace.tar', Path(*parts[3:]).as_posix()


def manifest_entries(manifest: dict) -> tuple[list[tuple[str, dict]], list[str]]:
    """Read both the original path-map and the later sorted-entry manifest."""
    if 'files' not in manifest:
        return [], ['files inventory is missing']
    files = manifest['files']
    if isinstance(files, dict):
        if not files:
            return [], ['files inventory must not be empty']
        return list(files.items()), []
    if not isinstance(files, list):
        return [], ['files must be an object or array']
    if not files:
        return [], ['files inventory must not be empty']
    entries: list[tuple[str, dict]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            errors.append(f'files[{index}] must be an object')
            continue
        relative = item.get('path')
        if not isinstance(relative, str) or not relative:
            errors.append(f'files[{index}].path must be a nonempty string')
            continue
        if relative in seen:
            errors.append(f'duplicate manifest path: {relative}')
            continue
        seen.add(relative)
        entries.append((relative, item))
    return entries, errors


def verify(evidence: Path) -> tuple[int, list[str]]:
    root = evidence.resolve()
    manifest_path = root / 'evidence-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entries, failures = manifest_entries(manifest)
    verified = 0
    for relative, expected in entries:
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            failures.append(relative)
            continue
        path = root / relative_path
        actual: tuple[str, int] | None = None
        if is_safe_regular_file(path, root):
            with path.open('rb') as stream:
                actual = digest_stream(stream)
        else:
            archive_entry = workspace_archive_entry(root, relative)
            if archive_entry is not None:
                archive, member_name = archive_entry
                if is_safe_regular_file(archive, root):
                    try:
                        with tarfile.open(archive, 'r:') as bundle:
                            member = bundle.getmember(member_name)
                            if member.isfile():
                                archive_stream = bundle.extractfile(member)
                                if archive_stream is not None:
                                    with archive_stream:
                                        actual = digest_stream(archive_stream)
                    except (KeyError, tarfile.TarError, OSError):
                        actual = None
        expected_size = expected.get('bytes')
        expected_hash = expected.get('sha256')
        if (type(expected_size) is not int or expected_size < 0
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in '0123456789abcdef' for character in expected_hash)):
            failures.append(relative)
        elif actual != (expected_hash, expected_size):
            failures.append(relative)
        else:
            verified += 1
    return verified, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path, help='Completed run directory containing evidence-manifest.json')
    args = parser.parse_args()
    if not args.evidence.is_dir():
        parser.error(f'evidence directory does not exist: {args.evidence}')
    manifest_path = args.evidence / 'evidence-manifest.json'
    if not manifest_path.is_file():
        parser.error(f'evidence manifest does not exist: {manifest_path}')
    verified, failures = verify(args.evidence)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f'manifest_sha256={manifest_sha}')
    print(f'verified_files={verified}')
    if failures:
        print(f'failed_files={len(failures)}')
        for relative in failures:
            print(f'FAIL {relative}')
        return 1
    print('status=COMPLETED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
