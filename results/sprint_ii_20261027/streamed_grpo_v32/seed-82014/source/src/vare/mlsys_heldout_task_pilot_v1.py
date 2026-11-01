"""Narrow task source and patch controls for the frozen two-task pilot."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v1.json"
LOCK_PATH = ROOT / "protocols/mlsys_heldout_task_qualification_pilot_v1.lock.json"
_DIFF_PATH = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")


class QualificationError(ValueError):
    """Task identity or source qualification failed closed."""


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes())


def write_exclusive_atomic(path: Path, raw: bytes) -> str:
    """Durably publish complete bytes without replacing an existing artifact."""
    if path.exists() or path.is_symlink():
        raise QualificationError(f"append-only artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return sha256(raw)


def load_frozen_protocol(
    protocol_path: Path = PROTOCOL_PATH,
    lock_path: Path = LOCK_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != protocol.get("protocol_id"):
        raise QualificationError("protocol identity mismatch")
    if lock.get("protocol_sha256") != sha256(protocol_raw):
        raise QualificationError("protocol hash changed after freeze")
    catalog = ROOT / "results/mlsys-task-deep-qualification-v1/qualification.json"
    allocation = ROOT / "results/mlsys-task-deep-qualification-v1/protocol.json"
    if sha256_file(catalog) != lock.get("source_catalog_sha256"):
        raise QualificationError("frozen source catalog hash mismatch")
    if sha256_file(allocation) != lock.get("allocation_sha256"):
        raise QualificationError("frozen allocation hash mismatch")
    catalog_json = json.loads(catalog.read_text(encoding="utf-8"))
    allocation_json = json.loads(allocation.read_text(encoding="utf-8"))
    selected = {task["task_id"]: task for task in protocol["task_contracts"]}
    original_tasks = {task["task_id"]: task for task in catalog_json["tasks"]}
    for task_id, contract in selected.items():
        original = original_tasks.get(task_id)
        if original is None or original.get("split") != "heldout":
            raise QualificationError(f"task is not in the frozen held-out pool: {task_id}")
        if task_id not in allocation_json["split"]["heldout"]:
            raise QualificationError(f"held-out allocation mismatch: {task_id}")
        for field, value in (
            ("repository", contract["repository"]),
            ("baseline_revision", contract["baseline_commit"]),
            ("fixed_revision", contract["fixed_commit"]),
            ("grader_path", contract["grader_path"]),
            ("grader_selector", contract["grader_selector"]),
            ("grader_revision", contract["grader_revision"]),
        ):
            if original.get(field) != value:
                raise QualificationError(f"frozen source metadata mismatch: {task_id}/{field}")
        prompt = ROOT / contract["candidate_prompt_path"]
        if sha256_file(prompt) != contract["candidate_prompt_sha256"]:
            raise QualificationError(f"task prompt hash mismatch: {task_id}")
    return protocol, lock


def git_bytes(repo: Path, revision: str, path: str) -> bytes:
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.useReplaceRefs=false", "show", f"{revision}:{path}"],
        env=env,
        timeout=30,
    )


def verify_task_source(repo: Path, contract: dict[str, Any]) -> dict[str, Any]:
    source = {}
    for name, revision in (
        ("baseline", contract["baseline_commit"]),
        ("fixed", contract["fixed_commit"]),
    ):
        subprocess.run(
            ["git", "-C", str(repo), "-c", "core.useReplaceRefs=false", "cat-file", "-e", f"{revision}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=10,
            env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
        )
        target = git_bytes(repo, revision, contract["target_path"])
        expected = contract[f"{name}_target_sha256"]
        if sha256(target) != expected:
            raise QualificationError(f"{name} target hash mismatch for {contract['task_id']}")
        source[name] = {"revision": revision, "target_sha256": sha256(target), "target_bytes": target}
    grader = git_bytes(repo, contract["grader_revision"], contract["grader_path"])
    if sha256(grader) != contract["grader_sha256"]:
        raise QualificationError(f"grader source hash mismatch for {contract['task_id']}")
    source["grader"] = {"revision": contract["grader_revision"], "sha256": sha256(grader), "bytes": grader}
    if contract.get("license_file_path"):
        license_file = git_bytes(repo, contract["fixed_commit"], contract["license_file_path"])
        if sha256(license_file) != contract.get("license_file_sha256"):
            raise QualificationError(f"license source hash mismatch for {contract['task_id']}")
        source["license"] = {
            "path": contract["license_file_path"],
            "revision": contract["fixed_commit"],
            "sha256": sha256(license_file),
        }
    return source


def validate_patch(patch: str, allowed_target: str) -> list[str]:
    """Accept a single regular-file patch for exactly one allowlisted target."""
    if not isinstance(patch, str) or not patch or len(patch.encode("utf-8")) > 256_000:
        raise QualificationError("patch is empty or over the size cap")
    if "GIT binary patch" in patch:
        raise QualificationError("binary patches are forbidden")
    paths: list[str] = []
    headers: list[tuple[str, str]] = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            fields = line.split()
            if len(fields) != 4:
                raise QualificationError("malformed diff header")
            left = _DIFF_PATH.fullmatch(fields[2])
            right = _DIFF_PATH.fullmatch(fields[3])
            if left is None or right is None:
                raise QualificationError("unsafe diff header path")
            headers.append((left.group(1), right.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            raw_path = line[4:].split("\t", 1)[0]
            match = _DIFF_PATH.fullmatch(raw_path)
            if match is None:
                raise QualificationError("unsafe file path")
            path = PurePosixPath(match.group(1))
            if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                raise QualificationError("path traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(("rename from ", "rename to ", "new file mode ", "deleted file mode ")):
            raise QualificationError("file creation, deletion, and rename are forbidden")
    if headers != [(allowed_target, allowed_target)] or paths != [allowed_target, allowed_target]:
        raise QualificationError("patch must edit only the task target")
    return paths


def unified_target_patch(target: str, before: bytes, after: bytes) -> str:
    try:
        old = before.decode("utf-8").splitlines(keepends=True)
        new = after.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise QualificationError("target source must be UTF-8 text") from exc
    body = "".join(difflib.unified_diff(old, new, fromfile=f"a/{target}", tofile=f"b/{target}"))
    if not body:
        return ""
    return f"diff --git a/{target} b/{target}\n{body}"


def python_package_blobs(repo: Path, revision: str) -> dict[str, bytes]:
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.useReplaceRefs=false", "ls-tree", "-rz", "--full-tree", "-r", revision, "--", "mlx_lm"],
        env=env,
        timeout=30,
    )
    files: dict[str, bytes] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, path_raw = item.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        path = path_raw.decode("utf-8")
        if not path.startswith("mlx_lm/"):
            raise QualificationError("source tree escaped the package prefix")
        if mode == "120000" or object_type != "blob":
            raise QualificationError(f"symlink or non-blob source rejected: {path}")
        if path.endswith(".py"):
            files[path] = subprocess.check_output(
                ["git", "-C", str(repo), "-c", "core.useReplaceRefs=false", "cat-file", "blob", object_id],
                env=env,
                timeout=30,
            )
    if not files:
        raise QualificationError("pinned source has no Python package files")
    return files


def materialize_candidate(
    source_files: dict[str, bytes],
    target: str,
    target_bytes: bytes,
    adapter_source: Path,
    destination: Path,
) -> dict[str, str]:
    if destination.exists() or destination.is_symlink():
        raise QualificationError("candidate destination must be new")
    if target not in source_files or not target.startswith("mlx_lm/"):
        raise QualificationError("target is outside the pinned source package")
    destination.mkdir(parents=True)
    file_hashes: dict[str, str] = {}
    for relative, content in sorted(source_files.items()):
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(target_bytes if relative == target else content)
        path.chmod(0o444)
        file_hashes[relative] = sha256(target_bytes if relative == target else content)
    file_hashes[target] = sha256(target_bytes)
    adapter = destination / "vare_task_adapter.py"
    adapter.write_bytes(adapter_source.read_bytes())
    adapter.chmod(0o444)
    file_hashes["vare_task_adapter.py"] = sha256_file(adapter)
    verify_candidate_surface(destination, file_hashes)
    return file_hashes


def mutate_target(task_id: str, fixed: bytes) -> bytes:
    mutations = {
        "mlxlm-1902-continuation-score-truncation": (
            b"completion_start = len(prefix)\n",
            b"completion_start = len(prefix) - 1\n",
        ),
        "mlxlm-1926-falcon-cache-gradient": (
            b"cache = [(None, None)] * len(self.layers)\n",
            b"cache = [(None, None)] * max(len(self.layers) - 1, 0)\n",
        ),
        "mlxlm-1938-phi3small-logits-mask": (
            b"out[..., self._dummy_tokenizer_ids] = -float(\"inf\")\n",
            b"out[self._dummy_tokenizer_ids] = -float(\"inf\")\n",
        ),
        "mlxlm-1979-ssm-padding-mask": (
            b"dtA = dtA * mask[..., None]\n",
            b"dtA = dtA\n",
        ),
    }
    try:
        before, after = mutations[task_id]
    except KeyError as exc:
        raise QualificationError(f"no prespecified mutant for {task_id}") from exc
    if fixed.count(before) != 1:
        raise QualificationError(f"mutant anchor mismatch for {task_id}")
    return fixed.replace(before, after, 1)


def package_delta(base: dict[str, bytes], candidate: dict[str, str]) -> list[str]:
    paths = sorted(set(base) | (set(candidate) - {"vare_task_adapter.py"}))
    changed = []
    for path in paths:
        if path not in base or path not in candidate or sha256(base[path]) != candidate[path]:
            changed.append(path)
    return changed


def verify_candidate_surface(root: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise QualificationError("candidate namespace is not a real directory")
    actual: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise QualificationError("candidate namespace contains a symlink")
        if path.is_dir() and path.name in {".git", "tests", "evaluator", "__pycache__"}:
            raise QualificationError("candidate namespace includes protected assets")
        if path.is_file():
            if path.suffix != ".py":
                raise QualificationError("candidate namespace includes a non-source file")
            actual[path.relative_to(root).as_posix()] = sha256_file(path)
    if actual != expected_hashes:
        raise QualificationError("candidate source tree or hash changed after staging")
    return actual

