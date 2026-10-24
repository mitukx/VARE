"""Read-only source, artifact, metadata, and cache inventory for one image."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys


TARGET = Path("/usr/local/lib/python3.12/site-packages/mlx_lm/models/mimo_v2_flash.py")
FIXED_SHA256 = "a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_code(code):
    yield code
    for value in code.co_consts:
        if hasattr(value, "co_consts") and hasattr(value, "co_names"):
            yield from walk_code(value)


def main() -> None:
    dist = metadata.distribution("mlx-lm")
    target_bytes = TARGET.read_bytes() if TARGET.is_file() else b""
    pyc_paths = sorted(TARGET.parent.joinpath("__pycache__").glob("mimo_v2_flash*.pyc"))
    pyc_report = []
    for path in pyc_paths:
        data = path.read_bytes()
        names = []
        try:
            import marshal

            code = marshal.loads(data[16:])
            names = sorted({name for obj in walk_code(code) for name in obj.co_names})
        except Exception as exc:
            names = [f"unreadable:{type(exc).__name__}"]
        pyc_report.append({
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": len(data),
            "contains_value_scale_symbol": "attention_value_scale" in names,
            "contains_value_scale_constant": b"attention_value_scale" in data,
            "magic_hex": data[:4].hex(),
        })
    record_matches = []
    record_path = Path(dist._path) / "RECORD"
    if record_path.exists():
        record_matches = [
            row for row in record_path.read_text(encoding="utf-8").splitlines()
            if "mimo_v2_flash" in row
        ]
    scanned_roots = [Path("/usr/local/lib/python3.12/site-packages"), Path("/root/.cache"), Path("/tmp"), Path("/opt")]
    exact_source_paths = []
    wheel_paths = []
    pyc_symbol_paths = []
    for root in scanned_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.suffix == ".py" and sha256(path) == FIXED_SHA256:
                    exact_source_paths.append(str(path))
                if path.suffix == ".whl":
                    wheel_paths.append(str(path))
                if path.suffix == ".pyc" and b"attention_value_scale" in path.read_bytes():
                    pyc_symbol_paths.append(str(path))
            except (OSError, PermissionError):
                continue
    result = {
        "python": sys.version,
        "distribution": "mlx-lm",
        "distribution_version": dist.version,
        "distribution_metadata_dir": str(dist._path),
        "requires_dist": dist.requires or [],
        "target_source_path": str(TARGET),
        "target_source_sha256": sha256(TARGET) if target_bytes else None,
        "target_source_size_bytes": len(target_bytes),
        "target_source_matches_pinned_fix": bool(target_bytes) and sha256(TARGET) == FIXED_SHA256,
        "target_pyc_files": pyc_report,
        "record_entries_for_target": record_matches,
        "exact_fixed_source_paths_in_scanned_roots": sorted(exact_source_paths),
        "pyc_files_with_value_scale_symbol_in_scanned_roots": sorted(pyc_symbol_paths),
        "wheel_files_in_scanned_roots": sorted(wheel_paths),
        "scan_roots": [str(root) for root in scanned_roots],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
