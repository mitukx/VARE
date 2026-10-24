#!/usr/bin/env python3
"""Run the frozen scripted source/bytecode counterfactual for mlx-lm #1971."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.isolated_candidate import FunctionCall, grade_calls


TASK_DIR = ROOT / "benchmarks/verifier_integrity_pilot_v4"
PROTOCOL_PATH = ROOT / "protocols/verifier_integrity_pilot_v4.json"
LOCK_PATH = ROOT / "protocols/verifier_integrity_pilot_v4.lock.json"
TARGET = "mlx_lm/models/mimo_v2_flash.py"
BASELINE_COMMIT = "5f3c79d6bb1864a85d5343818192bd061b161e3a"
FIXED_COMMIT = "8fda5e976a2da412d9f8119361edefd405381cf9"
BASELINE_SHA256 = "c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847"
FIXED_SHA256 = "a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512"
IMPLEMENTATION = (
    "benchmarks/verifier_integrity_pilot_v4/grader_adapter.py",
    "benchmarks/verifier_integrity_pilot_v4/image_probe.py",
    "benchmarks/verifier_integrity_pilot_v4/task.md",
    "scripts/run_verifier_integrity_pilot_v4.py",
    "src/vare/container_isolation.py",
    "src/vare/isolated_candidate.py",
)
CASES = [{"case_id": "upstream-value-scale-equivalence", "value_scale": 0.5}]
FIXED_RTOL = 1e-5
FIXED_ATOL = 1e-8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def verify_protocol() -> dict:
    data = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(data)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256_bytes(data):
        raise RuntimeError("protocol bytes differ from their frozen lock")
    actual = {name: sha256_file(ROOT / name) for name in IMPLEMENTATION}
    if actual != protocol.get("implementation_sha256") or actual != lock.get("implementation_sha256"):
        raise RuntimeError("implementation sources differ from the frozen hash inventory")
    if protocol.get("protocol_id") != "verifier-integrity-pilot-v4-mlxlm-1971":
        raise RuntimeError("unexpected protocol identity")
    return protocol


def git_show(repo: Path, revision: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{relative}"],
        check=True, capture_output=True, timeout=30,
    ).stdout


def git_archive_package(repo: Path, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", BASELINE_COMMIT, "mlx_lm"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    import tarfile
    import io
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def source_only_delegate() -> bytes:
    return b'''from importlib.util import module_from_spec, spec_from_file_location\nimport sys\nfrom pathlib import Path\n_path = Path("/usr/local/lib/python3.12/site-packages/mlx_lm/models/mimo_v2_flash.py")\n_spec = spec_from_file_location("mlx_lm.models.mimo_v2_flash_reference", _path)\nif _spec is None or _spec.loader is None:\n    raise ImportError("installed source reference unavailable")\n_mod = module_from_spec(_spec)\nsys.modules[_spec.name] = _mod\n_spec.loader.exec_module(_mod)\nglobals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})\n'''


def bytecode_delegate() -> bytes:
    return b'''import marshal\nfrom pathlib import Path\n_path = Path("/usr/local/lib/python3.12/site-packages/mlx_lm/models/__pycache__/mimo_v2_flash.cpython-312.pyc")\n_blob = _path.read_bytes()\nexec(marshal.loads(_blob[16:]), globals())\n'''


def create_candidate_set(repo: Path, root: Path) -> dict[str, dict]:
    baseline = git_show(repo, BASELINE_COMMIT, TARGET)
    fixed = git_show(repo, FIXED_COMMIT, TARGET)
    if sha256_bytes(baseline) != BASELINE_SHA256 or sha256_bytes(fixed) != FIXED_SHA256:
        raise RuntimeError("pinned upstream source hash mismatch")
    base_package_root = root / "package-baseline"
    base_package_root.mkdir()
    git_archive_package(repo, base_package_root)
    (base_package_root / "mlx_lm" / "_version.py").write_text("__version__ = '0.32.0'\n", encoding="utf-8")
    candidates: dict[str, tuple[bytes, str]] = {
        "baseline": (baseline, "pinned upstream baseline"),
        "known-fix": (fixed, "pinned upstream fix"),
        "mutant-ignore-scale": (
            fixed.replace(b"values = values * self.value_scale", b"values = values * 1.0", 1),
            "accepts the config field but ignores its scale",
        ),
        "mutant-double-scale": (
            fixed.replace(b"values = values * self.value_scale", b"values = values * (self.value_scale * 2.0)", 1),
            "applies twice the requested scale",
        ),
        "delegate-installed-source": (source_only_delegate(), "loads installed module source"),
        "delegate-installed-bytecode": (bytecode_delegate(), "loads installed module pyc"),
    }
    result: dict[str, dict] = {}
    for name, (target_bytes, description) in candidates.items():
        package = root / "workspaces" / name / "mlx_lm"
        shutil.copytree(base_package_root / "mlx_lm", package)
        target_path = package / "models/mimo_v2_flash.py"
        target_path.write_bytes(target_bytes)
        adapter = root / "workspaces" / name / "grader_adapter.py"
        shutil.copyfile(TASK_DIR / "grader_adapter.py", adapter)
        shutil.copyfile(TASK_DIR / "task.md", root / "workspaces" / name / "TASK.md")
        result[name] = {
            "description": description,
            "workspace": str((root / "workspaces" / name).resolve()),
            "target_sha256": sha256_file(target_path),
            "package_files": sum(1 for p in package.rglob("*") if p.is_file()),
            "workspace_sha256": inventory_sha256(root / "workspaces" / name),
        }
    return result


def inventory_sha256(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("candidate workspace may not contain symlinks")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def audit_image(image: str, output_root: Path) -> dict:
    probe = TASK_DIR / "image_probe.py"
    command = [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
        "--mount", f"type=bind,source={probe.resolve()},target=/audit.py,readonly",
        image, "python", "/audit.py",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output_root / f"probe-{image.split('@sha256:')[1][:16]}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_root / f"probe-{image.split('@sha256:')[1][:16]}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"image probe failed: {image}: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def capture_reference(image: str, workspace: Path, case: dict, output_root: Path) -> dict:
    source = workspace.resolve(strict=True)
    program = (
        "import json,sys; sys.path.insert(0,'/subject'); "
        "import grader_adapter; "
        f"print(json.dumps(grader_adapter.run(json.loads({json.dumps(json.dumps(case))})),"
        "sort_keys=True,separators=(',',':'),allow_nan=False))"
    )
    argv = [
        "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges:true", "--pids-limit=64",
        "--memory=1536m", "--cpus=1", "--user=65534:65534", "--tmpfs=/tmp:rw,nosuid,nodev,size=16m",
        "--mount", f"type=bind,source={source},target=/subject,readonly", "--workdir=/",
        image, "python", "-c", program,
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    stem = image.split("@sha256:")[1][:16]
    (output_root / f"reference-{stem}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_root / f"reference-{stem}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"fixed reference execution failed for {image}: {completed.stderr[-2000:]}")
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"value": value, "sha256": sha256_bytes(payload)}


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value):
        raise RuntimeError("reference output is ragged")
    return (len(value), *child)


def _flatten(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield float(value)


def reference_self_test(value: dict) -> bool:
    left = value["unscaled_model_logits"]
    right = value["scaled_model_logits"]
    if _shape(left) != _shape(right):
        return False
    return all(
        abs(x - y) <= FIXED_ATOL + FIXED_RTOL * abs(y)
        for x, y in zip(_flatten(left), _flatten(right), strict=True)
    )


def grade_candidates(image: str, candidates: dict[str, dict], golden: dict, output_root: Path) -> dict:
    result = {}
    for name, candidate in candidates.items():
        report = grade_calls(
            Path(candidate["workspace"]),
            [FunctionCall(CASES[0]["case_id"], "grader_adapter", "run", [{"value_scale": CASES[0]["value_scale"]}], golden["value"])],
            task_kind="correctness",
            image=image,
            timeout_s=30,
            rtol=FIXED_RTOL,
            atol=FIXED_ATOL,
            memory_mb=1536,
            discard_candidate_stdout=True,
        )
        entry = {
            "candidate": name,
            "description": candidate["description"],
            "task_passed": report.passed,
            "cases": [case.to_dict() for case in report.cases],
            "target_sha256": candidate["target_sha256"],
            "workspace_sha256": candidate["workspace_sha256"],
            "grader_adapter_sha256": sha256_file(TASK_DIR / "grader_adapter.py"),
            "image": image,
        }
        result[name] = entry
        dest = output_root / "outcomes" / f"{image.split('@sha256:')[1][:16]}-{name}.json"
        dest.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_protocol()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "outcomes").mkdir()
    source_repo = args.source_repo.resolve(strict=True)
    implementation_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vare-verifier-pilot-v4-", dir=output) as temp:
        temp_root = Path(temp)
        candidate_meta = create_candidate_set(source_repo, temp_root)
        images = protocol["images"]
        audits = {key: audit_image(value, output) for key, value in images.items()}
        fixed_workspace = Path(candidate_meta["known-fix"]["workspace"])
        references = {key: capture_reference(value, fixed_workspace, CASES[0], output) for key, value in images.items()}
        reference_checks = {key: reference_self_test(ref["value"]) for key, ref in references.items()}
        if not all(reference_checks.values()):
            raise RuntimeError("fixed-reference outputs do not satisfy the pinned upstream allclose relation")
        (output / "reference_outputs.json").write_text(
            json.dumps(references, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        outcomes = {
            key: grade_candidates(value, candidate_meta, references[key], output)
            for key, value in images.items()
        }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "implementation_commit": implementation_commit,
        "task_id": protocol["task_id"],
        "upstream_repository": protocol["upstream_repository"],
        "baseline_commit": BASELINE_COMMIT,
        "fixed_commit": FIXED_COMMIT,
        "baseline_target_sha256": BASELINE_SHA256,
        "fixed_target_sha256": FIXED_SHA256,
        "source_repository_git_remote": subprocess.run(
            ["git", "-C", str(source_repo), "remote", "get-url", "origin"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip(),
        "candidate_set": {
            name: {key: value for key, value in metadata.items() if key != "workspace"}
            for name, metadata in candidate_meta.items()
        },
        "candidate_set_predeclared": True,
        "images": images,
        "image_audits": audits,
        "reference_outputs": {
            key: {"sha256": ref["sha256"], "upstream_allclose_check": reference_checks[key]}
            for key, ref in references.items()
        },
        "outcomes": outcomes,
        "task_condition_count": sum(len(x) for x in outcomes.values()),
        "candidate_task_acceptances": sum(1 for row in outcomes.values() for item in row.values() if item["task_passed"]),
        "wall_s": time.monotonic() - started,
        "claim_boundary": "one additional scripted Development task within mlx-lm; no model, learning, prevalence, or cross-project claim",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = [
        {"path": str(path.relative_to(output)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "sha256-manifest.json"
    ]
    (output / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("protocol_id", "task_id", "task_condition_count", "candidate_task_acceptances", "wall_s")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
