#!/usr/bin/env python3
"""Independent, offline, source-bound grader for the MLX-LM MoE task."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, TypeGuard, cast


BASELINE = "77c33b14373ac70d7abd6f82af15962852adadbb"
EDITABLE = (
    "mlx_lm/models/granitemoe.py",
    "mlx_lm/models/granitemoehybrid.py",
    "mlx_lm/models/lfm2_moe.py",
)
TEST_SELECTORS = (
    "TestTunerTrainer::test_granitemoe_backward",
    "TestTunerTrainer::test_granitemoehybrid_backward",
    "TestTunerTrainer::test_lfm2_moe_backward",
)
TOKENS = [[1, 2, 3, 4]]
WORKER_SEED = 20261103
MODEL_CASES: dict[str, dict[str, Any]] = {
    "granitemoe": {
        "model_type": "granitemoe",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "logits_scaling": 1.0,
        "attention_multiplier": 1.0,
        "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0,
        "max_position_embeddings": 128,
        "num_key_value_heads": 2,
        "attention_bias": False,
        "rope_theta": 10_000.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
    },
    "granitemoehybrid": {
        "model_type": "granitemoehybrid",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "attention_bias": False,
        "embedding_multiplier": 1.0,
        "attention_multiplier": 1.0,
        "logits_scaling": 1.0,
        "residual_multiplier": 1.0,
        "layer_types": ["attention"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "shared_intermediate_size": 32,
    },
    "lfm2_moe": {
        "model_type": "lfm2_moe",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "use_expert_bias": False,
        "num_dense_layers": 1,
        "norm_eps": 1e-6,
        "conv_bias": False,
        "conv_L_cache": 3,
        "layer_types": ["conv", "full_attention"],
    },
}
GRADIENT_PATHS = {
    "granitemoe": ("switch_mlp", "router.layer"),
    "granitemoehybrid": ("switch_mlp", "router.layer"),
    "lfm2_moe": ("switch_mlp", "feed_forward.gate"),
}
ROUTING_CLASSES = {
    "granitemoe": "GraniteMoeTopKGating",
    "granitemoehybrid": "GraniteMoeHybridTopKGating",
    "lfm2_moe": "Lfm2MoeSparseMoeBlock",
}
MAX_WALL_SECONDS = 240
MAX_RSS_BYTES = 8 * 1024**3
MAX_OUTPUT_BYTES = 1_048_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_sha256(root: Path) -> str:
    """Hash a directory's relative paths, file contents and symlink targets."""
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            item = b"L\0" + os.readlink(path).encode("utf-8")
        elif path.is_file():
            item = b"F\0" + bytes.fromhex(sha256_file(path))
        elif path.is_dir():
            item = b"D\0"
        else:
            raise ValueError(f"unsupported filesystem object in source tree: {path}")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item)
    return digest.hexdigest()


def _load_lock(task_root: Path) -> tuple[dict[str, Any], str]:
    grader_file = Path(__file__).resolve()
    lock_path = task_root / "protocol.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if sha256_file(grader_file) != lock["grader"]["sha256"]:
        raise ValueError("independent grader source hash mismatch")
    worker_file = task_root / "evaluator" / "candidate_worker.py"
    if sha256_file(worker_file) != lock["grader"]["worker_sha256"]:
        raise ValueError("candidate worker source hash mismatch")
    if lock["source"]["baseline_revision"] != BASELINE:
        raise ValueError("protocol baseline revision mismatch")
    if tuple(lock["grader"]["selectors"]) != TEST_SELECTORS:
        raise ValueError("protocol test-selector mismatch")
    repository_root = task_root.parents[2]
    runner = (repository_root / lock["calibration"]["runner_path"]).resolve(strict=True)
    if repository_root not in runner.parents:
        raise ValueError("calibration runner path escapes the VARE repository")
    if sha256_file(runner) != lock["calibration"]["runner_sha256"]:
        raise ValueError("calibration runner source hash mismatch")
    return lock, sha256_file(lock_path)


def _assert_task_eligible(lock: dict[str, Any]) -> None:
    if lock.get("status") == "POST_EXPOSURE_SECURITY_CORRECTION":
        raise RuntimeError(
            "this task was consumed before its grader isolation correction; "
            "calibration and candidate grading require a fresh task identity"
        )


def _git(workspace: Path, *args: str) -> str:
    git = "/usr/bin/git" if Path("/usr/bin/git").exists() else shutil.which("git")
    if not git:
        raise RuntimeError("git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    result = subprocess.run(
        [
            git,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(workspace),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=30,
    )
    return result.stdout.strip()


def changed_paths(workspace: Path) -> tuple[str, ...]:
    tracked = _git(
        workspace,
        "diff",
        "--name-only",
        "--no-ext-diff",
        "--no-renames",
        BASELINE,
        "--",
    )
    untracked = _git(workspace, "ls-files", "--others", "--exclude-standard")
    paths = {path for path in (*tracked.splitlines(), *untracked.splitlines()) if path}
    return tuple(sorted(paths))


def _all_links(root: Path) -> list[str]:
    links: list[str] = []
    for path in root.rglob("*"):
        if path == root / ".git" or root / ".git" in path.parents:
            continue
        if path.is_symlink():
            links.append(path.relative_to(root).as_posix())
    return sorted(links)


def _validate_changed_paths(paths: tuple[str, ...]) -> None:
    if not set(paths).issubset(set(EDITABLE)):
        raise ValueError(f"candidate changed paths outside editable allowlist: {paths}")


def _validate_regular_source(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"candidate source is missing or not a regular file: {path}")
    if path.stat().st_nlink != 1:
        raise ValueError(f"candidate source must not be hard-linked: {path}")
    return sha256_file(path)


def _quote(path: Path) -> str:
    value = str(path.resolve())
    if any(char in value for char in ('"', "\\", "\n", "\r")):
        raise ValueError("sandbox path contains unsupported characters")
    return json.dumps(value)


def _path_filter(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved.is_dir():
        return f"(subpath {_quote(resolved)})"
    return f"(literal {_quote(resolved)})"


def _metal_cache_paths() -> tuple[Path, ...]:
    host_tmp = Path(tempfile.gettempdir()).resolve()
    cache_root = host_tmp.parent / "C"
    return tuple(
        path
        for path in (
            cache_root / "com.apple.metalfe",
            cache_root / "com.apple.gpuarchiver",
        )
        if path.is_dir()
    )


def _installed_mlx_lm_paths(
    runtime_root: Path, python_prefix: Path, python_base_prefix: Path
) -> tuple[Path, ...]:
    """Locate installed MLX-LM source, metadata, archives, and bytecode roots."""
    roots = {
        runtime_root,
        *(Path(value) for value in sys.path if "site-packages" in value),
    }
    for prefix in (python_prefix, python_base_prefix):
        roots.update(prefix.glob("lib/python*/site-packages"))
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in list(directory_names):
                path = current_path / name
                normalized = name.lower()
                if normalized == "mlx_lm" or normalized.startswith(
                    ("mlx_lm-", "mlx-lm")
                ):
                    if path.is_symlink():
                        raise ValueError(f"runtime MLX-LM path is symlinked: {path}")
                    found.add(path.resolve(strict=True))
                    directory_names.remove(name)
            for name in file_names:
                normalized = name.lower()
                if normalized.startswith(("mlx_lm-", "mlx-lm")) and normalized.endswith(
                    (".whl", ".zip", ".egg")
                ):
                    path = current_path / name
                    if path.is_symlink():
                        raise ValueError(f"runtime MLX-LM archive is symlinked: {path}")
                    found.add(path.resolve(strict=True))
    return tuple(sorted(found))


def sandbox_profile(
    *,
    trusted_source: Path,
    runtime_root: Path,
    scratch: Path,
    python: Path,
    python_prefix: Path,
    python_base_prefix: Path,
    forbidden_paths: tuple[Path, ...],
    metal_cache_paths: tuple[Path, ...],
) -> str:
    lines = [
        "(version 1)",
        "(deny default)",
        "(deny network*)",
    ]
    lines.extend(
        [
            "(allow sysctl-read)",
            f"(allow process-exec {_path_filter(python)})",
            f"(allow process-exec {_path_filter(python.resolve(strict=True))})",
            '(allow file-read* (literal "/dev/null"))',
            '(allow file-write* (literal "/dev/null"))',
            '(allow file-read* (literal "/dev/urandom"))',
            # dyld needs metadata for the filesystem root while resolving Python.
            # This is a read of the root directory itself, not a recursive grant.
            '(allow file-read-data (literal "/"))',
            '(allow iokit-open (iokit-user-client-class "AGXDeviceUserClient") '
            '(iokit-user-client-class "IOSurfaceRootUserClient"))',
            '(allow mach-lookup (global-name "com.apple.MTLCompilerService") '
            '(xpc-service-name "com.apple.MTLCompilerService"))',
        ]
    )
    for path in (
        Path("/System"),
        Path("/usr/bin"),
        Path("/usr/lib"),
        Path("/bin"),
        Path("/sbin"),
        python_prefix,
        python_base_prefix,
        runtime_root,
        trusted_source,
        scratch,
    ):
        if path.exists():
            lines.append(f"(allow file-read* (subpath {_quote(path)}))")
    lines.append(f"(allow file-write* (subpath {_quote(scratch)}))")
    for cache_path in metal_cache_paths:
        lines.append(
            "(allow file-issue-extension (require-all "
            '(extension-class "com.apple.app-sandbox.read-write") '
            f"(subpath {_quote(cache_path)})))"
        )
    # In Seatbelt profiles these later denials take precedence over a broader
    # allow (for example site-packages). Keep this block last and regression
    # test it against real sandbox-exec rather than relying on textual checks.
    for path in forbidden_paths:
        if path.exists():
            lines.append(f"(deny file-read* {_path_filter(path)})")
            lines.append(f"(deny file-write* {_path_filter(path)})")
    return "\n".join(lines) + "\n"


def process_tree_rss(root_pid: int) -> int:
    output = subprocess.check_output(["/bin/ps", "-axo", "pid=,ppid=,rss="], text=True)
    rows: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, ppid, rss_kib = map(int, fields)
        rows[pid] = rss_kib * 1024
        children.setdefault(ppid, []).append(pid)
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rows.get(pid, 0)
        pending.extend(children.get(pid, []))
    return total


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_output(pipe, chunks: list[bytes], state: dict[str, Any]) -> None:
    total = 0
    while True:
        block = pipe.read(65536)
        if not block:
            break
        remaining = MAX_OUTPUT_BYTES - total
        if remaining > 0:
            chunks.append(block[:remaining])
        total += len(block)
        if total > MAX_OUTPUT_BYTES:
            state["overflow"] = True
    state["total_bytes"] = total


def _parse_worker_report(
    output: bytes,
    *,
    expected_model: str,
    expected_nonce: str,
    expects_gradients: bool,
) -> dict[str, Any]:
    try:
        report = json.loads(output.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("candidate worker did not return one JSON report") from exc
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("candidate worker report schema mismatch")
    expected_keys = {
        "schema_version",
        "model",
        "nonce",
        "module_origin",
        "logits",
        "logits_finite",
    }
    if expects_gradients:
        expected_keys.update({"loss", "loss_finite", "gradients"})
    if set(report) != expected_keys:
        raise ValueError("candidate worker report fields mismatch")
    if report.get("model") != expected_model:
        raise ValueError("candidate worker returned a different model case")
    if report.get("nonce") != expected_nonce:
        raise ValueError("candidate worker report is not bound to this request")
    if not isinstance(report.get("module_origin"), str):
        raise ValueError("candidate worker omitted the imported module origin")
    if not isinstance(report.get("logits"), list):
        raise ValueError("candidate worker omitted logits")
    if type(report.get("logits_finite")) is not bool:
        raise ValueError("candidate worker omitted the logits finiteness flag")
    if expects_gradients:
        if type(report.get("loss_finite")) is not bool:
            raise ValueError("candidate worker omitted the loss finiteness flag")
        if (
            isinstance(report.get("loss"), bool)
            or not isinstance(report.get("loss"), (int, float))
            or not math.isfinite(report["loss"])
        ):
            raise ValueError("candidate worker returned an invalid loss")
        gradients = report.get("gradients")
        if not isinstance(gradients, list) or not gradients:
            raise ValueError("candidate worker returned no gradients")
        for row in gradients:
            if not isinstance(row, dict) or set(row) != {
                "name",
                "shape",
                "finite",
                "nonzero_count",
                "max_abs",
            }:
                raise ValueError("candidate worker returned a malformed gradient row")
            if (
                not isinstance(row["name"], str)
                or not isinstance(row["shape"], list)
                or any(type(size) is not int or size < 0 for size in row["shape"])
                or type(row["finite"]) is not bool
                or type(row["nonzero_count"]) is not int
                or row["nonzero_count"] < 0
                or isinstance(row["max_abs"], bool)
                or not isinstance(row["max_abs"], (int, float))
                or not math.isfinite(row["max_abs"])
                or row["max_abs"] < 0
            ):
                raise ValueError("candidate worker returned invalid gradient data")
    return report


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_numeric_tree(item) for item in value)
    return False


def _numeric_tree_close(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _numeric_tree_close(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _candidate_report_failures(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    model: str,
    expected_origin: Path,
) -> list[str]:
    failures: list[str] = []
    if Path(candidate["module_origin"]).resolve() != expected_origin.resolve():
        failures.append("candidate_module_loaded_from_unexpected_path")
    if not candidate.get("logits_finite") or not _finite_numeric_tree(
        candidate["logits"]
    ):
        failures.append("candidate_logits_nonfinite")
    if not _numeric_tree_close(candidate["logits"], reference["logits"]):
        failures.append("candidate_forward_differs_from_pinned_baseline")
    if not candidate.get("loss_finite") or not isinstance(
        candidate.get("loss"), (int, float)
    ):
        failures.append("candidate_loss_nonfinite_or_missing")
    gradients = candidate.get("gradients")
    if not isinstance(gradients, list) or not gradients:
        failures.append("candidate_gradients_missing")
        return failures
    if any(not isinstance(row, dict) or not row.get("finite") for row in gradients):
        failures.append("candidate_gradient_nonfinite")
    expert_marker, router_marker = GRADIENT_PATHS[model]
    if not any(
        expert_marker in str(row.get("name", ""))
        and row.get("nonzero_count", 0) > 0
        and row.get("max_abs", 0.0) > 0.0
        for row in gradients
        if isinstance(row, dict)
    ):
        failures.append("selected_expert_gradient_missing")
    if not any(
        router_marker in str(row.get("name", ""))
        and row.get("nonzero_count", 0) > 0
        and row.get("max_abs", 0.0) > 0.0
        for row in gradients
        if isinstance(row, dict)
    ):
        failures.append("router_gradient_missing")
    return failures


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in node.elts))
    return set()


def _is_mx_call(node: ast.AST | None, name: str) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mx"
    )


def _contains_mx_call(node: ast.AST | None, name: str) -> bool:
    return node is not None and any(_is_mx_call(item, name) for item in ast.walk(node))


class _RoutingFixNormalizer(ast.NodeTransformer):
    """Remove only the expected route-index detach before AST comparison."""

    def __init__(self, *, class_name: str, index_name: str) -> None:
        self.class_name = class_name
        self.index_name = index_name
        self.in_target_class = False
        self.in_target_method = False
        self.function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        prior = self.in_target_class
        self.in_target_class = node.name == self.class_name and not prior
        visited = cast(ast.ClassDef, self.generic_visit(node))
        self.in_target_class = prior
        return visited

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        prior = self.in_target_method
        self.in_target_method = (
            self.in_target_class
            and self.function_depth == 0
            and node.name == "__call__"
        )
        self.function_depth += 1
        visited = cast(ast.FunctionDef, self.generic_visit(node))
        self.function_depth -= 1
        self.in_target_method = prior
        return visited

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        prior = self.in_target_method
        self.in_target_method = (
            self.in_target_class
            and self.function_depth == 0
            and node.name == "__call__"
        )
        self.function_depth += 1
        visited = cast(ast.AsyncFunctionDef, self.generic_visit(node))
        self.function_depth -= 1
        self.in_target_method = prior
        return visited

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if not self.in_target_method:
            return self.generic_visit(node)
        names = set().union(*(_target_names(target) for target in node.targets))
        if (
            self.index_name == "top_k_idx"
            and self.index_name in names
            and _contains_mx_call(node.value, "argpartition")
            and _is_mx_call(node.value, "stop_gradient")
            and node.value.args
        ):
            node.value = node.value.args[0]
        elif (
            self.index_name == "inds"
            and self.index_name in names
            and _is_mx_call(node.value, "stop_gradient")
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "inds"
        ):
            return None
        return self.generic_visit(node)


def _routing_fix_failures(
    source_root: Path, baseline_root: Path
) -> dict[str, list[str]]:
    """Check the intended detach and reject unrelated source behavior changes."""
    results: dict[str, list[str]] = {}
    for model_name in MODEL_CASES:
        source_path = source_root / "mlx_lm" / "models" / f"{model_name}.py"
        baseline_path = baseline_root / "mlx_lm" / "models" / f"{model_name}.py"
        try:
            source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            baseline_tree = ast.parse(baseline_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            results[model_name] = ["model_source_unreadable_or_invalid"]
            continue
        failures = ["routing_indices_not_detached_at_selection"]
        for function in (
            method
            for class_node in ast.walk(source_tree)
            if isinstance(class_node, ast.ClassDef)
            and class_node.name == ROUTING_CLASSES[model_name]
            for method in class_node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and method.name == "__call__"
        ):
            nodes = list(ast.walk(function))
            take_uses = [
                (node.lineno, node.args[1].id)
                for node in nodes
                if _is_mx_call(node, "take_along_axis")
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Name)
            ]
            if model_name == "lfm2_moe":
                index_names = {"inds"}
                partition_assignments = [
                    node
                    for node in nodes
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and _contains_mx_call(node.value, "argpartition")
                    and any(
                        name in index_names
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                        for name in _target_names(target)
                    )
                ]
                stop_assignments = [
                    node
                    for node in nodes
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(
                        "inds" in _target_names(target)
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    )
                    and _is_mx_call(node.value, "stop_gradient")
                    and node.value.args
                    and isinstance(node.value.args[0], ast.Name)
                    and node.value.args[0].id == "inds"
                ]
                if partition_assignments and stop_assignments and take_uses:
                    stop_line = min(node.lineno for node in stop_assignments)
                    if all(
                        node.lineno < stop_line for node in partition_assignments
                    ) and any(
                        stop_line < line and name == "inds" for line, name in take_uses
                    ):
                        failures = []
                        break
            else:
                index_name = "top_k_idx"
                partition_assignments = [
                    node
                    for node in nodes
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and _contains_mx_call(node.value, "argpartition")
                    and any(
                        index_name in _target_names(target)
                        for target in (
                            node.targets
                            if isinstance(node, ast.Assign)
                            else [node.target]
                        )
                    )
                ]
                detached = [
                    node
                    for node in partition_assignments
                    if _is_mx_call(node.value, "stop_gradient")
                    and _contains_mx_call(node.value, "argpartition")
                ]
                if (
                    partition_assignments
                    and len(detached) == len(partition_assignments)
                    and take_uses
                    and any(
                        node.lineno < line and name == index_name
                        for node in detached
                        for line, name in take_uses
                    )
                ):
                    failures = []
                    break
        if not failures:
            index_name = "inds" if model_name == "lfm2_moe" else "top_k_idx"
            normalized_source = _RoutingFixNormalizer(
                class_name=ROUTING_CLASSES[model_name], index_name=index_name
            ).visit(source_tree)
            normalized_baseline = _RoutingFixNormalizer(
                class_name=ROUTING_CLASSES[model_name], index_name=index_name
            ).visit(baseline_tree)
            if ast.dump(normalized_source, include_attributes=False) != ast.dump(
                normalized_baseline, include_attributes=False
            ):
                failures = ["source_modified_outside_routing_fix"]
        results[model_name] = failures
    return results


def _run_worker(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    request: dict[str, Any],
) -> dict[str, Any]:
    payload = (
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("candidate worker request exceeds the protocol limit")
    stdout_chunks: list[bytes] = []
    reader_state: dict[str, Any] = {}
    started = time.monotonic()
    peak_rss = 0
    timed_out = False
    rss_exceeded = False
    output_exceeded = False
    monitor_error = None
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    reader = threading.Thread(
        target=_read_output,
        args=(proc.stdout, stdout_chunks, reader_state),
        daemon=True,
    )
    reader.start()
    try:
        proc.stdin.write(payload)
        proc.stdin.close()
    except (BrokenPipeError, OSError) as exc:
        monitor_error = f"worker_request_write:{type(exc).__name__}: {exc}"
        _terminate(proc)
    while proc.poll() is None:
        try:
            peak_rss = max(peak_rss, process_tree_rss(proc.pid))
        except Exception as exc:
            monitor_error = f"{type(exc).__name__}: {exc}"
            _terminate(proc)
            break
        if reader_state.get("overflow"):
            output_exceeded = True
            _terminate(proc)
            break
        if peak_rss > MAX_RSS_BYTES:
            rss_exceeded = True
            _terminate(proc)
            break
        if time.monotonic() - started > MAX_WALL_SECONDS:
            timed_out = True
            _terminate(proc)
            break
        time.sleep(0.2)
    proc.wait()
    _kill_process_group(proc)
    reader.join(timeout=5)
    return {
        "returncode": proc.returncode,
        "output": b"".join(stdout_chunks),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "timed_out": timed_out,
        "rss_limit_exceeded": rss_exceeded,
        "output_limit_exceeded": output_exceeded,
        "monitor_error": monitor_error,
    }


def _runtime_versions(python: Path, pythonpath: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json,platform,sys; "
        "names={'pytest':'pytest','mlx':'mlx','mlx_lm':'mlx-lm','numpy':'numpy'}; "
        "out={'python':platform.python_version()}; "
        "[out.__setitem__(k,m.version(v)) for k,v in names.items()]; "
        "print(json.dumps(out,sort_keys=True))"
    )
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": str(pythonpath),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    return json.loads(result.stdout)


def _validate_candidate(
    workspace: Path,
    expected_base_hashes: dict[str, str],
    *,
    allow_unmodified_baseline: bool,
) -> tuple[tuple[str, ...], dict[str, str]]:
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError(
            "macOS sandbox-exec is unavailable; refusing to grade source"
        )
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("candidate workspace must be a real directory")
    if (workspace / ".git").is_symlink() or not (workspace / ".git").is_dir():
        raise ValueError("candidate git metadata is missing or symlinked")
    if _git(workspace, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("candidate workspace HEAD is not the pinned baseline")
    if _git(workspace, "rev-parse", "--is-shallow-repository") != "true":
        raise ValueError("candidate workspace is not shallow")
    if _git(workspace, "remote"):
        raise ValueError("candidate workspace still has a Git remote")
    paths = changed_paths(workspace)
    if not paths and not allow_unmodified_baseline:
        raise ValueError("candidate contains no source changes")
    _validate_changed_paths(paths)
    if _all_links(workspace):
        raise ValueError("candidate workspace contains symlinks outside Git metadata")
    hashes: dict[str, str] = {}
    for relative in EDITABLE:
        path = workspace / relative
        hashes[relative] = _validate_regular_source(path)
        if relative not in paths and hashes[relative] != expected_base_hashes[relative]:
            raise ValueError(f"unreported candidate source change: {relative}")
    return paths, hashes


def _write_profile(profile_path: Path, text: str) -> None:
    profile_path.write_text(text, encoding="utf-8")


def grade_candidate(
    *,
    task_root: Path,
    source_manifest_path: Path,
    workspace: Path,
    trusted_baseline: Path,
    grader_test: Path,
    python: Path,
    pythonpath: Path,
    output_dir: Path,
    expected_arm: str = "candidate",
) -> dict[str, Any]:
    task_root = task_root.resolve(strict=True)
    source_manifest_path = source_manifest_path.resolve(strict=True)
    if workspace.expanduser().is_symlink():
        raise ValueError("candidate workspace must not be a symlink")
    workspace = workspace.expanduser().resolve(strict=True)
    trusted_baseline = trusted_baseline.resolve(strict=True)
    grader_test = grader_test.resolve(strict=True)
    # Keep the venv entrypoint path. It is commonly a symlink whose prefix is
    # meaningful to Python; resolving it silently changes the interpreter env.
    python = python.expanduser()
    if not python.is_file():
        raise ValueError("pinned Python interpreter is missing")
    pythonpath = pythonpath.resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite grader evidence: {output_dir}")

    lock, lock_hash = _load_lock(task_root)
    _assert_task_eligible(lock)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["baseline_revision"] != BASELINE:
        raise ValueError("prepared source manifest baseline mismatch")
    if source_manifest["protocol_lock_sha256"] != lock_hash:
        raise ValueError(
            "prepared source manifest is bound to a different grader protocol"
        )
    if (
        source_manifest["task_json_sha256"]
        != lock["task_definition"]["task_json_sha256"]
    ):
        raise ValueError("prepared source manifest task.json hash mismatch")
    if source_manifest["task_md_sha256"] != lock["task_definition"]["task_md_sha256"]:
        raise ValueError("prepared source manifest TASK.md hash mismatch")
    if source_manifest["candidate_visible_files"] != list(EDITABLE):
        raise ValueError("prepared candidate allowlist mismatch")
    if source_manifest["trusted_baseline_source"] != str(trusted_baseline):
        raise ValueError("trusted baseline path does not match prepared manifest")
    if source_manifest["trusted_grader_test"] != str(grader_test):
        raise ValueError("trusted grader test path does not match prepared manifest")
    if source_manifest["grader_test_sha256"] != lock["grader"]["upstream_test_sha256"]:
        raise ValueError("prepared source manifest test hash mismatch")
    if source_manifest["license_sha256"] != lock["source"]["license_file_sha256"]:
        raise ValueError("prepared source manifest license hash mismatch")
    if (
        source_manifest["baseline_file_sha256"]
        != lock["source"]["baseline_file_sha256"]
    ):
        raise ValueError("prepared baseline file hashes differ from the protocol lock")
    if sha256_file(grader_test) != lock["grader"]["upstream_test_sha256"]:
        raise ValueError("trusted grader test hash mismatch")
    if (grader_test.parent / ".git").exists():
        raise ValueError("grader test directory contains Git metadata")
    if _git(trusted_baseline, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("trusted source tree is not the pinned baseline")
    if _git(trusted_baseline, "remote"):
        raise ValueError("trusted source tree still has a Git remote")
    if _git(trusted_baseline, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("trusted source tree is not clean")
    if (
        sha256_file(trusted_baseline / "LICENSE")
        != lock["source"]["license_file_sha256"]
    ):
        raise ValueError("trusted source license hash mismatch")
    if _git(workspace, "rev-parse", "HEAD") != BASELINE:
        raise ValueError("candidate source tree is not the pinned baseline")
    expected_base_hashes = lock["source"]["baseline_file_sha256"]
    for relative, expected_hash in expected_base_hashes.items():
        if sha256_file(trusted_baseline / relative) != expected_hash:
            raise ValueError(f"trusted baseline file hash mismatch: {relative}")

    paths, candidate_hashes = _validate_candidate(
        workspace,
        expected_base_hashes,
        allow_unmodified_baseline=expected_arm == "baseline",
    )
    protected_inputs = (
        task_root,
        source_manifest_path,
        trusted_baseline,
        grader_test,
        python.resolve(),
        pythonpath,
    )
    if any(path == workspace or workspace in path.parents for path in protected_inputs):
        raise ValueError(
            "a trusted grader input or runtime is inside the candidate workspace"
        )
    if output_dir == workspace or workspace in output_dir.parents:
        raise ValueError(
            "grader evidence output must be outside the candidate workspace"
        )

    versions = _runtime_versions(python, pythonpath)
    expected_versions = lock["runtime"]
    if versions["python"] != expected_versions["python"]:
        raise RuntimeError(f"Python runtime mismatch: {versions}")
    if (
        versions["mlx"] != expected_versions["mlx"]
        or versions["mlx_lm"] != expected_versions["mlx_lm"]
    ):
        raise RuntimeError(f"MLX runtime mismatch: {versions}")

    output_dir.mkdir(parents=True)
    scratch_root = output_dir / "sandbox-scratch"
    scratch_root.mkdir()
    candidate_root = output_dir / "candidate-source"
    reference_root = output_dir / "reference-source"
    worker_source = task_root / "evaluator" / "candidate_worker.py"
    ignore_caches = shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.pyo", "tests", "test_*"
    )
    shutil.copytree(
        trusted_baseline / "mlx_lm", candidate_root / "mlx_lm", ignore=ignore_caches
    )
    shutil.copytree(
        trusted_baseline / "mlx_lm", reference_root / "mlx_lm", ignore=ignore_caches
    )
    for relative in EDITABLE:
        destination = candidate_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / relative, destination)
    shutil.copyfile(worker_source, candidate_root / "vare_candidate_worker.py")
    shutil.copyfile(worker_source, reference_root / "vare_candidate_worker.py")
    for root in (candidate_root, reference_root):
        for item in root.rglob("*"):
            if item.is_file():
                item.chmod(0o444)
            elif item.is_dir():
                item.chmod(0o555)
        root.chmod(0o555)

    python_prefix = Path(sys.prefix)
    python_base_prefix = Path(sys.base_prefix)
    installed_mlx_lm = _installed_mlx_lm_paths(
        pythonpath, python_prefix, python_base_prefix
    )
    known_fix_root = trusted_baseline.parent / "trusted-known-fix"
    reference_forbidden = (
        task_root,
        source_manifest_path,
        workspace,
        grader_test,
        grader_test.parent,
        known_fix_root,
        *installed_mlx_lm,
    )
    candidate_forbidden = (
        *reference_forbidden,
        trusted_baseline,
        reference_root,
    )
    cache_paths = _metal_cache_paths()
    common_env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_NO_INDEX": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }

    def run_reference(
        model_name: str, model_args: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        scratch = scratch_root / f"reference-{model_name}"
        for name in ("home", "tmp"):
            (scratch / name).mkdir(parents=True, exist_ok=True)
        profile_path = output_dir / f"reference-sandbox-{model_name}.sb"
        profile = sandbox_profile(
            trusted_source=reference_root,
            runtime_root=pythonpath,
            scratch=scratch,
            python=python,
            python_prefix=python_prefix,
            python_base_prefix=python_base_prefix,
            forbidden_paths=reference_forbidden,
            metal_cache_paths=cache_paths,
        )
        _write_profile(profile_path, profile)
        env = {
            **common_env,
            "HOME": str(scratch / "home"),
            "TMPDIR": str(scratch / "tmp"),
            "PYTHONPATH": os.pathsep.join((str(reference_root), str(pythonpath))),
        }
        request: dict[str, Any] = {
            "model": model_name,
            "nonce": secrets.token_hex(32),
            "model_args": model_args,
            "tokens": TOKENS,
            "seed": WORKER_SEED,
            "compute_gradients": False,
        }
        run = _run_worker(
            [
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile_path),
                str(python),
                "-S",
                str(reference_root / "vare_candidate_worker.py"),
            ],
            cwd=reference_root,
            environment=env,
            request=request,
        )
        raw_path = output_dir / f"reference-worker-{model_name}.log"
        raw_path.write_bytes(run["output"])
        if (
            run["returncode"] != 0
            or run["timed_out"]
            or run["rss_limit_exceeded"]
            or run["output_limit_exceeded"]
            or run["monitor_error"]
        ):
            raise RuntimeError(f"pinned baseline worker failed for {model_name}: {run}")
        report = _parse_worker_report(
            run["output"],
            expected_model=model_name,
            expected_nonce=request["nonce"],
            expects_gradients=False,
        )
        expected_origin = (
            reference_root / "mlx_lm" / "models" / f"{model_name}.py"
        ).resolve()
        if Path(report["module_origin"]).resolve() != expected_origin:
            raise RuntimeError(
                f"pinned baseline imported from an unexpected path: {model_name}"
            )
        if not report.get("logits_finite") or not _finite_numeric_tree(
            report["logits"]
        ):
            raise RuntimeError(
                f"pinned baseline produced nonfinite logits: {model_name}"
            )
        receipt = {
            "model": model_name,
            "returncode": run["returncode"],
            "elapsed_seconds": run["elapsed_seconds"],
            "sampled_process_tree_peak_rss_bytes": run[
                "sampled_process_tree_peak_rss_bytes"
            ],
            "output_sha256": sha256_file(raw_path),
            "sandbox_profile_sha256": sha256_file(profile_path),
        }
        return report, receipt

    reference_reports: dict[str, dict[str, Any]] = {}
    reference_receipts: list[dict[str, Any]] = []
    for model_name, model_args in MODEL_CASES.items():
        report, receipt = run_reference(model_name, model_args)
        reference_reports[model_name] = report
        reference_receipts.append(receipt)

    source_structure = _routing_fix_failures(candidate_root, trusted_baseline)
    case_results: list[dict[str, Any]] = []
    candidate_receipts: list[dict[str, Any]] = []
    infrastructure_ok = True
    started = time.monotonic()
    peak_rss = 0
    for model_name, model_args in MODEL_CASES.items():
        scratch = scratch_root / f"candidate-{model_name}"
        for name in ("home", "tmp"):
            (scratch / name).mkdir(parents=True, exist_ok=True)
        profile_path = output_dir / f"candidate-sandbox-{model_name}.sb"
        profile = sandbox_profile(
            trusted_source=candidate_root,
            runtime_root=pythonpath,
            scratch=scratch,
            python=python,
            python_prefix=python_prefix,
            python_base_prefix=python_base_prefix,
            forbidden_paths=candidate_forbidden,
            metal_cache_paths=cache_paths,
        )
        _write_profile(profile_path, profile)
        env = {
            **common_env,
            "HOME": str(scratch / "home"),
            "TMPDIR": str(scratch / "tmp"),
            "PYTHONPATH": os.pathsep.join((str(candidate_root), str(pythonpath))),
        }
        request: dict[str, Any] = {
            "model": model_name,
            "nonce": secrets.token_hex(32),
            "model_args": model_args,
            "tokens": TOKENS,
            "seed": WORKER_SEED,
            "compute_gradients": True,
        }
        run = _run_worker(
            [
                "/usr/bin/sandbox-exec",
                "-f",
                str(profile_path),
                str(python),
                "-S",
                str(candidate_root / "vare_candidate_worker.py"),
            ],
            cwd=candidate_root,
            environment=env,
            request=request,
        )
        raw_path = output_dir / f"candidate-worker-{model_name}.log"
        raw_path.write_bytes(run["output"])
        peak_rss = max(peak_rss, run["sampled_process_tree_peak_rss_bytes"])
        infrastructure_ok = infrastructure_ok and not (
            run["timed_out"]
            or run["rss_limit_exceeded"]
            or run["output_limit_exceeded"]
            or run["monitor_error"]
        )
        receipt = {
            "model": model_name,
            "returncode": run["returncode"],
            "elapsed_seconds": run["elapsed_seconds"],
            "sampled_process_tree_peak_rss_bytes": run[
                "sampled_process_tree_peak_rss_bytes"
            ],
            "timed_out": run["timed_out"],
            "rss_limit_exceeded": run["rss_limit_exceeded"],
            "output_limit_exceeded": run["output_limit_exceeded"],
            "monitor_error": run["monitor_error"],
            "output_sha256": sha256_file(raw_path),
            "sandbox_profile_sha256": sha256_file(profile_path),
        }
        candidate_receipts.append(receipt)
        failures: list[str]
        candidate_report: dict[str, Any] | None = None
        try:
            if run["returncode"] != 0:
                raise ValueError(f"candidate worker exited with {run['returncode']}")
            candidate_report = _parse_worker_report(
                run["output"],
                expected_model=model_name,
                expected_nonce=request["nonce"],
                expects_gradients=True,
            )
            failures = _candidate_report_failures(
                candidate_report,
                reference_reports[model_name],
                model=model_name,
                expected_origin=candidate_root
                / "mlx_lm"
                / "models"
                / f"{model_name}.py",
            )
        except (ValueError, OSError) as exc:
            failures = [f"worker_protocol_or_execution_failure:{type(exc).__name__}"]
            if (
                b"Operation not permitted" in run["output"]
                or b"PermissionError" in run["output"]
            ):
                infrastructure_ok = False
        failures.extend(source_structure[model_name])
        case_results.append(
            {
                "model": model_name,
                "passed": not failures,
                "failures": failures,
                "source_structure_failures": source_structure[model_name],
                "candidate_report": candidate_report,
                "reference_logits_sha256": sha256_bytes(
                    json.dumps(
                        reference_reports[model_name]["logits"], separators=(",", ":")
                    ).encode("utf-8")
                ),
                "candidate_worker_output_sha256": sha256_file(raw_path),
                "elapsed_seconds": run["elapsed_seconds"],
                "sampled_process_tree_peak_rss_bytes": run[
                    "sampled_process_tree_peak_rss_bytes"
                ],
            }
        )

    passed = infrastructure_ok and all(case["passed"] for case in case_results)
    result = {
        "schema_version": 2,
        "task_id": lock["task_id"],
        "arm": expected_arm,
        "passed": passed,
        "grader_status": "PASS"
        if passed
        else ("TEST_FAIL" if infrastructure_ok else "INFRASTRUCTURE_FAIL"),
        "evaluation_mode": "static source contract plus isolated MLX worker and trusted parent comparison",
        "pytest_output_parser_used": False,
        "upstream_test_executed": False,
        "upstream_test_selectors_for_provenance": list(TEST_SELECTORS),
        "source_ast_delta": "only pinned top-k route-index detach edits are admitted",
        "worker_report_nonce_bound": True,
        "candidate_received_grader_tests": False,
        "candidate_received_reference_outputs": False,
        "candidate_original_workspace_visible": False,
        "candidate_write_paths": [
            str(scratch_root),
            *(str(path) for path in scratch_root.rglob("*") if path.is_dir()),
        ],
        "candidate_revision": _git(workspace, "rev-parse", "HEAD"),
        "changed_paths": list(paths),
        "candidate_source_sha256": candidate_hashes,
        "trusted_baseline_revision": _git(trusted_baseline, "rev-parse", "HEAD"),
        "upstream_test_source_sha256_not_executed": sha256_file(grader_test),
        "protocol_lock_sha256": lock_hash,
        "grader_sha256": sha256_file(Path(__file__).resolve()),
        "worker_sha256": sha256_file(worker_source),
        "python_sha256": sha256_file(python),
        "runtime_versions": versions,
        "model_cases": list(MODEL_CASES),
        "reference_runs": reference_receipts,
        "candidate_runs": candidate_receipts,
        "case_results": case_results,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sampled_process_tree_peak_rss_bytes": peak_rss,
        "process_tree_rss_limit_bytes": MAX_RSS_BYTES,
        "timed_out": any(run["timed_out"] for run in candidate_receipts),
        "rss_limit_exceeded": any(
            run["rss_limit_exceeded"] for run in candidate_receipts
        ),
        "output_limit_exceeded": any(
            run["output_limit_exceeded"] for run in candidate_receipts
        ),
        "monitor_error": next(
            (
                run["monitor_error"]
                for run in candidate_receipts
                if run["monitor_error"]
            ),
            None,
        ),
        "network_denied": True,
        "grader_test_outside_candidate_workspace": workspace not in grader_test.parents,
        "installed_mlx_lm_paths_denied": [str(path) for path in installed_mlx_lm],
    }
    (output_dir / "grade.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "task_id",
                    "arm",
                    "passed",
                    "grader_status",
                    "case_results",
                    "elapsed_seconds",
                    "sampled_process_tree_peak_rss_bytes",
                )
            },
            sort_keys=True,
        )
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--trusted-baseline", type=Path, required=True)
    parser.add_argument("--grader-test", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="candidate")
    args = parser.parse_args()
    try:
        result = grade_candidate(
            task_root=args.task_root,
            source_manifest_path=args.source_manifest,
            workspace=args.workspace,
            trusted_baseline=args.trusted_baseline,
            grader_test=args.grader_test,
            python=args.python,
            pythonpath=args.pythonpath,
            output_dir=args.output,
            expected_arm=args.arm,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "grader_status": "INFRASTRUCTURE_FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
