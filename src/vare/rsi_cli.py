"""Operator checks for the existing real-model coding RL path.

The preflight is intentionally read-only. It checks local runtime prerequisites
for the pinned MLX runner and does not assess task eligibility, run a model, or
authorize a training/evaluation experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
MODEL_EXECUTION_SCRIPTS = (
    "scripts/long_horizon_eval.py",
    "scripts/qwen3_function_repair_grpo.py",
)
GIB = 1024**3
DISK_MIN_GIB = 8.0
AVAILABLE_MEMORY_MIN_GIB = 8.0
MEMORY_FREE_MIN_PCT = 15.0
START_SWAP_MAX = 0.94


def _check(
    name: str, state: str, value: Any, detail: str, *, blocking: bool
) -> dict[str, Any]:
    return {
        "name": name,
        "state": state,
        "value": value,
        "detail": detail,
        "blocking": blocking,
    }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _hf_hub_cache() -> Path:
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _model_snapshot(model_path: str | None) -> tuple[Path, bool]:
    if model_path:
        path = Path(model_path).expanduser().resolve()
    else:
        path = (
            _hf_hub_cache()
            / f"models--{MODEL_ID.replace('/', '--')}"
            / "snapshots"
            / MODEL_REVISION
        )
    pinned_snapshot_layout = (
        path.name == MODEL_REVISION
        and path.parent.name == "snapshots"
        and path.parent.parent.name == f"models--{MODEL_ID.replace('/', '--')}"
    )
    weights_exist = path.is_dir() and any(
        next(path.glob(pattern), None) is not None
        for pattern in (
            "*.safetensors",
            "*.safetensors.index.json",
            "pytorch_model*.bin",
        )
    )
    return path, bool(
        pinned_snapshot_layout
        and path.is_dir()
        and (path / "config.json").is_file()
        and weights_exist
    )


def _physical_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _run_readonly(argv: list[str], timeout_s: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _available_memory_bytes() -> int | None:
    output = _run_readonly(["vm_stat"])
    if output is None:
        return None
    page_match = re.search(r"page size of (\d+) bytes", output)
    if page_match is None:
        return None
    pages = []
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        match = re.search(rf"^{re.escape(label)}:\s+(\d+)", output, flags=re.MULTILINE)
        if match is None:
            return None
        pages.append(int(match.group(1)))
    return int(page_match.group(1)) * sum(pages)


def _memory_free_pct() -> float | None:
    output = _run_readonly(["memory_pressure", "-Q"])
    if output is None:
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%", output)
    return float(match.group(1)) if match else None


def _swap_usage() -> dict[str, float] | None:
    output = _run_readonly(["sysctl", "-n", "vm.swapusage"])
    if output is None:
        return None
    match = re.search(r"total = ([\d.]+)M\s+used = ([\d.]+)M", output)
    if match is None:
        return None
    total_mib, used_mib = (float(value) for value in match.groups())
    total_gib, used_gib = total_mib / 1024, used_mib / 1024
    return {
        "total_gib": total_gib,
        "used_gib": used_gib,
        "fraction": used_gib / total_gib if total_gib else 0.0,
    }


def _docker_server_version(timeout_s: float = 5.0) -> tuple[str | None, str]:
    docker = shutil.which("docker")
    if docker is None:
        return None, "docker_cli_missing"
    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "docker_daemon_check_timed_out"
    except OSError:
        return None, "docker_cli_unavailable"
    if result.returncode != 0 or not result.stdout.strip():
        return None, "docker_daemon_unavailable"
    return result.stdout.strip(), "docker_daemon_available"


def build_preflight(
    *,
    repo_root: Path,
    run_root: Path,
    model_path: str | None = None,
    min_free_disk_gib: float = DISK_MIN_GIB,
) -> dict[str, Any]:
    """Return local runtime checks without importing model runtimes or loading weights."""
    if not math.isfinite(min_free_disk_gib) or min_free_disk_gib < 0:
        raise ValueError("--min-free-disk-gib must be finite and non-negative")

    repo_root = Path(repo_root).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        _check(
            "python_version",
            "pass" if python_ok else "blocked",
            platform.python_version(),
            "Python 3.11 以上が必要です" if not python_ok else "Python 3.11 以上です",
            blocking=True,
        )
    )

    apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }
    checks.append(
        _check(
            "mlx_platform",
            "pass" if apple_silicon else "blocked",
            {"system": platform.system(), "machine": platform.machine()},
            "固定したMLX実験はApple Siliconを対象にしています"
            if not apple_silicon
            else "Apple Silicon です",
            blocking=True,
        )
    )

    for module in ("mlx", "mlx_lm"):
        present = _module_available(module)
        checks.append(
            _check(
                f"python_module:{module}",
                "pass" if present else "blocked",
                present,
                "インストール済みです"
                if present
                else f"{module} はこの Python 環境にありません",
                blocking=True,
            )
        )

    _, model_ready = _model_snapshot(model_path)
    checks.append(
        _check(
            "pinned_model_snapshot",
            "pass" if model_ready else "blocked",
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "local_snapshot_present": model_ready,
            },
            "ローカルのモデルスナップショットを確認しました"
            if model_ready
            else "固定したモデルの重みがローカルにありません。プリフライトはダウンロードしません",
            blocking=True,
        )
    )

    missing_scripts = [
        name for name in MODEL_EXECUTION_SCRIPTS if not (repo_root / name).is_file()
    ]
    scripts_ready = not missing_scripts
    checks.append(
        _check(
            "execution_scripts",
            "pass" if scripts_ready else "blocked",
            {
                "present": len(MODEL_EXECUTION_SCRIPTS) - len(missing_scripts),
                "required": len(MODEL_EXECUTION_SCRIPTS),
            },
            "実行スクリプトをすべて確認しました"
            if scripts_ready
            else "不足ファイル: " + ", ".join(missing_scripts),
            blocking=True,
        )
    )

    disk_target = run_root
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    if not disk_target.is_dir():
        raise ValueError(
            "--run-root は既存ディレクトリ、またはその配下を指定してください"
        )
    disk = shutil.disk_usage(disk_target)
    min_free_bytes = int(min_free_disk_gib * GIB)
    disk_ready = disk.free >= min_free_bytes
    checks.append(
        _check(
            "free_disk",
            "pass" if disk_ready else "blocked",
            {"free_bytes": disk.free, "minimum_bytes": min_free_bytes},
            "空きディスクが設定した下限以上です"
            if disk_ready
            else "空きディスクが設定した下限を下回っています",
            blocking=True,
        )
    )

    available_memory = _available_memory_bytes()
    available_memory_gib = (
        available_memory / GIB if available_memory is not None else None
    )
    memory_ready = (
        available_memory_gib is None or available_memory_gib >= AVAILABLE_MEMORY_MIN_GIB
    )
    checks.append(
        _check(
            "available_memory",
            "unverified"
            if available_memory_gib is None
            else ("pass" if memory_ready else "blocked"),
            {
                "available_gib": available_memory_gib,
                "minimum_gib": AVAILABLE_MEMORY_MIN_GIB,
            },
            "空きメモリを測定できません"
            if available_memory_gib is None
            else (
                "空きメモリが実行スクリプトの下限以上です"
                if memory_ready
                else "空きメモリが実行スクリプトの下限を下回っています"
            ),
            blocking=available_memory_gib is not None,
        )
    )

    memory_free_pct = _memory_free_pct()
    pressure_ready = memory_free_pct is None or memory_free_pct >= MEMORY_FREE_MIN_PCT
    checks.append(
        _check(
            "memory_pressure",
            "unverified"
            if memory_free_pct is None
            else ("pass" if pressure_ready else "blocked"),
            {"free_pct": memory_free_pct, "minimum_pct": MEMORY_FREE_MIN_PCT},
            "メモリ空き率を測定できません"
            if memory_free_pct is None
            else (
                "メモリ空き率が実行スクリプトの下限以上です"
                if pressure_ready
                else "メモリ空き率が実行スクリプトの停止しきい値を下回っています"
            ),
            blocking=memory_free_pct is not None,
        )
    )

    swap = _swap_usage()
    swap_ready = swap is None or swap["fraction"] < START_SWAP_MAX
    checks.append(
        _check(
            "swap_pressure",
            "unverified" if swap is None else ("pass" if swap_ready else "blocked"),
            {**swap, "maximum_start_fraction": START_SWAP_MAX}
            if swap is not None
            else {
                "fraction": None,
                "maximum_start_fraction": START_SWAP_MAX,
            },
            "スワップ使用率を測定できません"
            if swap is None
            else (
                "スワップ使用率が実行スクリプトの開始しきい値未満です"
                if swap_ready
                else "スワップ使用率が実行スクリプトの開始しきい値以上です"
            ),
            blocking=swap is not None,
        )
    )

    docker_version, docker_reason = _docker_server_version()
    docker_ready = docker_version is not None
    checks.append(
        _check(
            "docker_candidate_worker",
            "pass" if docker_ready else "blocked",
            {"server_version": docker_version},
            {
                "docker_cli_missing": "docker CLI が見つかりません",
                "docker_cli_unavailable": "docker CLI を実行できません",
                "docker_daemon_check_timed_out": "Docker デーモンの確認がタイムアウトしました",
                "docker_daemon_unavailable": "Docker デーモンに接続できません",
                "docker_daemon_available": "Docker デーモンに接続できます",
            }.get(docker_reason, docker_reason),
            blocking=True,
        )
    )

    blockers = [
        item["name"] for item in checks if item["blocking"] and item["state"] != "pass"
    ]
    unverified = [item["name"] for item in checks if item["state"] == "unverified"]
    physical_memory = _physical_memory_bytes()
    runtime_status = (
        "blocked" if blockers else ("incomplete" if unverified else "ready")
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "local_runtime_only",
        "runtime_status": runtime_status,
        "blockers": blockers,
        "unverified_checks": unverified,
        "machine": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "physical_memory_bytes": physical_memory,
            "available_memory_bytes": available_memory,
            "memory_free_pct": memory_free_pct,
            "swap": swap,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_snapshot_present": model_ready,
            "explicit_model_path": bool(model_path),
        },
        "task_allocation": "not_assessed",
        "verifier_qualification": "not_assessed",
        "policy_update": "not_run",
        "capability_evaluation": "not_run",
        "model_loaded": False,
        "model_downloaded": False,
        "candidate_executed": False,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vare rsi",
        description="VARE の実モデル実験に向けた読み取り専用コマンドです。",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser(
        "preflight",
        help="モデルや候補コードを実行せずにローカル環境を確認します",
    )
    preflight.add_argument("--repo-root", type=Path, default=Path.cwd())
    preflight.add_argument("--run-root", type=Path, default=Path.cwd())
    preflight.add_argument(
        "--model-path",
        help="ローカルモデルのディレクトリを指定します。モデルIDからのダウンロードは行いません",
    )
    preflight.add_argument("--min-free-disk-gib", type=float, default=DISK_MIN_GIB)
    preflight.add_argument("--output", type=Path, help="JSON結果の保存先")
    return parser


def _write_json_atomic(path: Path, rendered: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_preflight(
        repo_root=args.repo_root,
        run_root=args.run_root,
        model_path=args.model_path,
        min_free_disk_gib=args.min_free_disk_gib,
    )
    rendered = (
        json.dumps(
            result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    if args.output:
        _write_json_atomic(args.output, rendered)
    print(rendered, end="")
    return 0 if result["runtime_status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
