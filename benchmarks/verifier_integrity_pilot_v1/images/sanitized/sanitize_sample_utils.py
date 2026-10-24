"""Replace the repair and remove installed bytecode and stale wheel caches."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path
import shutil


SITE = Path("/usr/local/lib/python3.12/site-packages")
PACKAGE = SITE / "mlx_lm"
TARGET = PACKAGE / "sample_utils.py"
BASELINE = Path("/tmp/vare-sanitize/sample_utils.py")
DIST_INFO = SITE / "mlx_lm-0.32.0.dist-info"
RECORD = DIST_INFO / "RECORD"
EXPECTED_INSTALLED = "3aac92d804b6b91c82a4e2844697a8723de26f6af57e10013a482b967e739cfa"
EXPECTED_BASELINE = "c93c1eef794725f9f7ce77b6212f61eb6d0fe17b9cd87c06cec6470ee12b07f2"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize() -> dict[str, object]:
    source = BASELINE.read_bytes()
    if digest(source) != EXPECTED_BASELINE:
        raise ValueError("pinned baseline source digest mismatch")
    if digest(TARGET.read_bytes()) != EXPECTED_INSTALLED:
        raise ValueError("installed source is not the expected contaminated version")
    if not RECORD.is_file() or len(list(SITE.glob("mlx_lm-*.dist-info/RECORD"))) != 1:
        raise ValueError("expected one installed mlx-lm RECORD")

    before = {
        str(path.relative_to(PACKAGE)): digest(path.read_bytes())
        for path in sorted(PACKAGE.rglob("*.py"))
    }
    shutil.copyfile(BASELINE, TARGET)
    bytecode = list(PACKAGE.rglob("*.pyc"))
    bytecode_bytes = sum(path.stat().st_size for path in bytecode)
    for path in bytecode:
        path.unlink()
    for path in sorted(PACKAGE.rglob("__pycache__"), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass

    with RECORD.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    relative_target = "mlx_lm/sample_utils.py"
    matching = [row for row in rows if row and row[0] == relative_target]
    if len(matching) != 1:
        raise ValueError("target must have exactly one RECORD entry")
    matching[0][1] = "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(source).digest()
    ).decode("ascii").rstrip("=")
    matching[0][2] = str(len(source))
    rows = [
        row
        for row in rows
        if not row
        or not row[0].endswith(".pyc")
        or (SITE / row[0]).is_file()
    ]
    with RECORD.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

    caches = []
    for path in (Path("/root/.cache/pip"), Path("/root/.cache/uv")):
        if path.exists():
            caches.append(str(path))
            shutil.rmtree(path)

    after = {
        str(path.relative_to(PACKAGE)): digest(path.read_bytes())
        for path in sorted(PACKAGE.rglob("*.py"))
    }
    expected_after = dict(before)
    expected_after["sample_utils.py"] = EXPECTED_BASELINE
    if after != expected_after:
        raise ValueError("Python source inventory changed beyond the target")
    if list(PACKAGE.rglob("*.pyc")):
        raise ValueError("installed MLX-LM bytecode remains")
    if digest(TARGET.read_bytes()) != EXPECTED_BASELINE:
        raise ValueError("baseline source replacement failed")
    return {
        "target_sha256": digest(TARGET.read_bytes()),
        "record_sha256": digest(RECORD.read_bytes()),
        "removed_bytecode_files": len(bytecode),
        "removed_bytecode_bytes": bytecode_bytes,
        "python_source_files_preserved": len(after),
        "wheel_caches_removed": caches,
    }


if __name__ == "__main__":
    print(json.dumps(sanitize(), sort_keys=True))
    shutil.rmtree("/tmp/vare-sanitize")
