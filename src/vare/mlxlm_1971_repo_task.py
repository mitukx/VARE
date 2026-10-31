"""Narrow isolated coding task for frozen MLX-LM #1971 behavior."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .isolated_candidate import FunctionCall, grade_calls

TASK_ID = "mlxlm-1971-mimo-value-scale"
BASELINE_COMMIT = "5f3c79d6bb1864a85d5343818192bd061b161e3a"
BASELINE_SOURCE_SHA256 = "c110f8deed605277ae070359dd6d2ba0c700981101240c54bcd049e8d60e4847"
FIXED_COMMIT = "8fda5e976a2da412d9f8119361edefd405381cf9"
FIXED_SOURCE_SHA256 = "a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512"
TARGET_PATH = "mlx_lm/models/mimo_v2_flash.py"
_PATCH_PATH = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")
TASK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
TASK_DESCRIPTION = TASK_ROOT / "task.md"
GRADER_ADAPTER = TASK_ROOT / "grader_adapter.py"
EXPECTED_OUTPUT = TASK_ROOT / "expected-output-v1.json"
IMAGE = "localhost:5000/vare/mlx-mimo-cpu-mimo@sha256:799f8b9f34fe2caa8fc0c4d0889abac63384606bcf40db28524db3fba07a7cbe"
MEMORY_MB = 1536
TIMEOUT_S = 30.0
CASE: dict[str, Any] = {
    "seed": 0,
    "value_scale": 0.5,
    "inputs": [[0, 1, 2, 3, 4, 5]],
    "config": {
        "model_type": "mimo_v2_flash",
        "num_experts_per_tok": 2,
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 0],
        "add_swa_attention_sink_bias": True,
        "add_full_attention_sink_bias": False,
        "sliding_window_size": 4,
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "n_shared_experts": None,
        "n_routed_experts": 4,
        "routed_scaling_factor": None,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "max_position_embeddings": 128,
        "layernorm_epsilon": 1e-5,
        "rope_theta": 1000.0,
        "swa_rope_theta": 1000.0,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "head_dim": 16,
        "v_head_dim": 8,
        "swa_head_dim": 16,
        "swa_v_head_dim": 8,
        "partial_rotary_factor": 0.5,
    },
}


class MlxLm1971TaskError(ValueError):
    """Pinned source or candidate patch did not meet the task contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(source_root: Path) -> None:
    root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not root.is_dir():
        raise MlxLm1971TaskError("pinned source root must be a real directory")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        flags = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", TARGET_PATH],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1971TaskError("source root is not a readable Git checkout") from exc
    target = root / TARGET_PATH
    if head != BASELINE_COMMIT or status:
        raise MlxLm1971TaskError("source checkout is not the clean pinned baseline")
    if target.is_symlink() or not target.is_file() or sha256_file(target) != BASELINE_SOURCE_SHA256:
        raise MlxLm1971TaskError("pinned baseline source SHA-256 mismatch")
    if any(line[:1] in {"h", "S"} for line in flags):
        raise MlxLm1971TaskError("target has assume-unchanged or skip-worktree flag")


def validate_patch(text: str) -> None:
    if not text or len(text.encode("utf-8")) > 256_000 or "GIT binary patch" in text:
        raise MlxLm1971TaskError("patch is empty, binary, or too large")
    headers: list[tuple[str, str]] = []
    paths: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            fields = line.split()
            if len(fields) != 4:
                raise MlxLm1971TaskError("malformed diff header")
            left, right = (_PATCH_PATH.fullmatch(fields[2]), _PATCH_PATH.fullmatch(fields[3]))
            if left is None or right is None:
                raise MlxLm1971TaskError("unsupported patch path")
            headers.append((left.group(1), right.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            match = _PATCH_PATH.fullmatch(line[4:].split("\t", 1)[0])
            if match is None:
                raise MlxLm1971TaskError("unsupported patch path")
            path = PurePosixPath(match.group(1))
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise MlxLm1971TaskError("patch traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(("rename from ", "rename to ", "new file mode ", "deleted file mode ")):
            raise MlxLm1971TaskError("file creation, deletion, and rename are forbidden")
    if headers != [(TARGET_PATH, TARGET_PATH)] or paths != [TARGET_PATH, TARGET_PATH]:
        raise MlxLm1971TaskError("patch must edit only mlx_lm/models/mimo_v2_flash.py")


def _copy_package(source_root: Path, destination: Path) -> None:
    package = source_root / "mlx_lm"
    if package.is_symlink() or not package.is_dir():
        raise MlxLm1971TaskError("pinned mlx_lm package is missing")
    for current, dirs, files in os.walk(package, followlinks=False):
        here = Path(current)
        dirs[:] = sorted(name for name in dirs if not (here / name).is_symlink())
        relative = here.relative_to(source_root)
        out = destination / relative
        out.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            src = here / name
            if src.is_symlink() or src.suffix != ".py":
                continue
            if relative == Path("mlx_lm") and name == "__init__.py":
                continue
            shutil.copyfile(src, out / name)


def export_candidate(source_root: Path, workspace: Path, patch_path: Path | None = None) -> dict[str, Any]:
    """Export from the fully inventoried v9 baseline implementation."""
    from .mlxlm_1971_repo_task_v9 import export_candidate_v9

    return export_candidate_v9(source_root, workspace, patch_path)


def _export_candidate_v1(source_root: Path, workspace: Path, patch_path: Path | None = None) -> dict[str, Any]:
    verify_baseline(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise MlxLm1971TaskError("candidate workspace already exists")
    if patch_path is not None:
        if patch_path.is_symlink() or not patch_path.is_file():
            raise MlxLm1971TaskError("patch must be a regular file")
        validate_patch(patch_path.read_text(encoding="utf-8"))
    workspace.mkdir(parents=True)
    _copy_package(source_root.resolve(), workspace)
    if patch_path is not None:
        try:
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, timeout=10)
            subprocess.run(["git", "apply", "--check", str(patch_path.resolve())], cwd=workspace, check=True, capture_output=True, timeout=10)
            subprocess.run(["git", "apply", str(patch_path.resolve())], cwd=workspace, check=True, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            shutil.rmtree(workspace, ignore_errors=True)
            raise MlxLm1971TaskError("candidate patch did not apply cleanly") from exc
        shutil.rmtree(workspace / ".git", ignore_errors=True)
    target = workspace / TARGET_PATH
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_patch_sha256": sha256_file(patch_path) if patch_path else None,
        "candidate_source_sha256": sha256_file(target),
        "fixed_source_match": sha256_file(target) == FIXED_SOURCE_SHA256,
        "changed_files": [TARGET_PATH] if patch_path else [],
        "git_history_exported": False,
        "tests_exported": False,
        "license_exported": False,
        "task_prompt_sha256": sha256_file(TASK_DESCRIPTION),
    }


def grade_workspace_v1(workspace: Path, *, image: str = IMAGE) -> dict[str, Any]:
    target = workspace / TARGET_PATH
    if workspace.is_symlink() or not workspace.is_dir() or target.is_symlink() or not target.is_file():
        raise MlxLm1971TaskError("candidate workspace is incomplete or unsafe")
    if image != IMAGE:
        raise MlxLm1971TaskError("requested image differs from the frozen protocol")
    if EXPECTED_OUTPUT.is_symlink() or not EXPECTED_OUTPUT.is_file():
        raise MlxLm1971TaskError("trusted expected output is missing")
    expected = json.loads(EXPECTED_OUTPUT.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-grade-") as folder:
        subject = Path(folder)
        _copy_package(workspace, subject)
        adapter = subject / "vare_task_adapter.py"
        shutil.copyfile(GRADER_ADAPTER, adapter)
        for path in (subject, *(subject / "mlx_lm").rglob("*.py"), adapter):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        report = grade_calls(
            subject,
            [FunctionCall("mimo-value-scale", "vare_task_adapter", "run", [CASE], expected)],
            task_kind="correctness",
            image=image,
            timeout_s=TIMEOUT_S,
            rtol=1e-5,
            atol=1e-8,
            memory_mb=MEMORY_MB,
            discard_candidate_stdout=True,
        )
    return {
        "task_id": TASK_ID,
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "candidate_source_sha256": sha256_file(target),
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "grader_adapter_sha256": sha256_file(GRADER_ADAPTER),
        "expected_output_sha256": sha256_file(EXPECTED_OUTPUT),
    }


def grade_workspace(workspace: Path, *, image: str = IMAGE) -> dict[str, Any]:
    """Grade under the current inventory-enforcing v9 protocol."""
    from .mlxlm_1971_repo_task_v9 import grade_workspace_v9

    return grade_workspace_v9(workspace, image=image)
