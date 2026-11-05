"""Reproducible experiment specifications for the execution backend.

A specification is data. It does not contain a trusted score, and loading it
does not execute candidate code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from vare.container_isolation import IsolationLimits

from .jsonutil import canonical_bytes, sha256_json, strict_load
from .paths import PathPolicyError, regular_outside, resolve_inside

_RUN_ID = __import__("re").compile(r"[A-Za-z0-9_-]{1,80}\Z")
_SECRET = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "AWS_", "OPENAI", "ANTHROPIC", "GITHUB", "GH_")


class SpecError(ValueError):
    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


@dataclass(frozen=True, slots=True)
class RunLimits:
    wall_time_s: float = 30.0
    cpu_time_s: int | None = 30
    memory_mb: int | None = 256
    max_output_bytes: int = 100_000
    max_file_bytes: int = 8_388_608
    max_child_processes: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.wall_time_s, bool) or not isinstance(self.wall_time_s, (int, float)):
            raise SpecError("invalid_spec", "wall_time_s must be a positive number")
        if not 0 < float(self.wall_time_s) <= 3600:
            raise SpecError("invalid_spec", "wall_time_s must be in (0, 3600]")
        if self.cpu_time_s is not None and (type(self.cpu_time_s) is not int or not 1 <= self.cpu_time_s <= 3600):
            raise SpecError("invalid_spec", "cpu_time_s must be an int in 1..3600 or null")
        if self.memory_mb is not None and (type(self.memory_mb) is not int or not 32 <= self.memory_mb <= 65536):
            raise SpecError("invalid_spec", "memory_mb must be an int in 32..65536 or null")
        if type(self.max_output_bytes) is not int or not 128 <= self.max_output_bytes <= 4_000_000:
            raise SpecError("invalid_spec", "max_output_bytes must be 128..4000000")
        if type(self.max_file_bytes) is not int or not 1024 <= self.max_file_bytes <= 64 * 1024 * 1024:
            raise SpecError("invalid_spec", "max_file_bytes must be 1024..67108864")
        if self.max_child_processes is not None and (
            type(self.max_child_processes) is not int or not 0 <= self.max_child_processes <= 256
        ):
            raise SpecError("invalid_spec", "max_child_processes must be an int in 0..256 or null")


@dataclass(frozen=True, slots=True)
class CommandStep:
    name: str
    argv: tuple[str, ...]
    cwd: str = "."

    def __post_init__(self) -> None:
        if not _RUN_ID.fullmatch(self.name):
            raise SpecError("invalid_spec", "step name must be a safe token")
        if not self.argv or not all(isinstance(item, str) and item and "\x00" not in item for item in self.argv):
            raise SpecError("invalid_spec", "argv must be a nonempty sequence of strings")
        if len(self.argv) > 64 or any(len(item) > 16_384 for item in self.argv):
            raise SpecError("invalid_spec", "argv is too large")
        raw = Path(self.cwd)
        if not self.cwd or raw.is_absolute() or ".." in raw.parts:
            raise SpecError("path_traversal", "step cwd must stay inside the workspace")


@dataclass(frozen=True, slots=True)
class MetricContract:
    """Trusted measurement contract. Candidate-authored success flags are not fields."""

    relative_path: str
    required_keys: tuple[str, ...]
    maxima: Mapping[str, float] = field(default_factory=dict)
    minima: Mapping[str, float] = field(default_factory=dict)
    timeout_counts_as_incorrect: bool = False

    def __post_init__(self) -> None:
        raw = Path(self.relative_path)
        if not self.relative_path or raw.is_absolute() or ".." in raw.parts:
            raise SpecError("path_traversal", "metric path escapes the workspace")
        if not self.required_keys or len(set(self.required_keys)) != len(self.required_keys):
            raise SpecError("invalid_spec", "metric contract needs unique required keys")
        if any(not isinstance(key, str) or not key for key in self.required_keys):
            raise SpecError("invalid_spec", "metric keys must be strings")
        required = set(self.required_keys)
        for label, bounds in (("maxima", self.maxima), ("minima", self.minima)):
            if not isinstance(bounds, Mapping):
                raise SpecError("invalid_spec", f"{label} must be a mapping")
            for key, value in bounds.items():
                if key not in required or isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SpecError("invalid_spec", f"{label} contains an untrusted bound")
        if not isinstance(self.timeout_counts_as_incorrect, bool):
            raise SpecError("invalid_spec", "timeout_counts_as_incorrect must be a bool")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    run_id: str
    task_id: str
    workspace: Path
    steps: tuple[CommandStep, ...]
    limits: RunLimits = field(default_factory=RunLimits)
    source_files: tuple[str, ...] = ()
    protected_paths: tuple[Path, ...] = ()
    metric_contract: MetricContract | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    backend: str = "cpu"
    isolation: str = "process_group"
    network_policy: str = "deny"
    docker_image: str | None = None
    allowed_executables: tuple[str, ...] = ()
    mlx_min_available_mb: int = 2048

    def __post_init__(self) -> None:
        if not _RUN_ID.fullmatch(self.run_id) or not _RUN_ID.fullmatch(self.task_id):
            raise SpecError("invalid_spec", "run_id and task_id must be safe tokens")
        if self.backend not in {"cpu", "mlx"}:
            raise SpecError("invalid_spec", "backend must be cpu or mlx")
        if self.isolation not in {"process_group", "docker"}:
            raise SpecError("invalid_spec", "isolation must be process_group or docker")
        if self.network_policy not in {"deny", "allow"}:
            raise SpecError("invalid_spec", "network_policy must be deny or allow")
        if self.isolation == "docker":
            if self.network_policy != "deny":
                raise SpecError("invalid_spec", "docker isolation always denies network")
            if not isinstance(self.docker_image, str):
                raise SpecError("isolation_unavailable", "docker isolation requires a digest-pinned image")
            # Reuse the existing image pin. This does not start a container.
            try:
                IsolationLimits(
                    image=self.docker_image,
                    memory_mb=self.limits.memory_mb or 512,
                    timeout_s=min(float(self.limits.wall_time_s), 3600),
                    max_output_bytes=min(self.limits.max_output_bytes, 512_000),
                )
            except ValueError as exc:
                raise SpecError("isolation_unavailable", str(exc)) from exc
        elif self.docker_image is not None:
            raise SpecError("invalid_spec", "docker_image is only valid for docker isolation")
        if not self.steps or len(self.steps) > 32:
            raise SpecError("invalid_spec", "an experiment needs 1..32 steps")
        if len({step.name for step in self.steps}) != len(self.steps):
            raise SpecError("invalid_spec", "step names must be unique")
        workspace = Path(self.workspace)
        if not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir():
            raise SpecError("path_traversal", "workspace must be an absolute real directory")
        object.__setattr__(self, "workspace", workspace.resolve())
        if len(set(self.source_files)) != len(self.source_files):
            raise SpecError("invalid_spec", "source files must be unique")
        for relative in self.source_files:
            resolve_inside(self.workspace, relative, expect="file")
        protected: list[Path] = []
        for raw in self.protected_paths:
            protected.append(regular_outside(Path(raw), self.workspace))
        object.__setattr__(self, "protected_paths", tuple(protected))
        if self.metric_contract is not None:
            # The metric file may be created by the run. Its parents must not be links.
            parent = Path(self.metric_contract.relative_path).parent
            if str(parent) not in {"", "."}:
                existing = self.workspace
                for part in parent.parts:
                    existing = existing / part
                    if existing.is_symlink():
                        raise PathPolicyError("symlink_escape", "metric parent is a symlink")
        _reject_env(self.env)
        if type(self.mlx_min_available_mb) is not int or not 256 <= self.mlx_min_available_mb <= 65536:
            raise SpecError("invalid_spec", "mlx_min_available_mb is out of range")
        allowed: list[str] = []
        for raw in self.allowed_executables:
            path = Path(raw)
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise SpecError("executable_not_allowed", "allowed executable must be an absolute regular file")
            if path.resolve() != path:
                raise SpecError("symlink_escape", "allowed executable path is not stable")
            allowed.append(str(path))
        object.__setattr__(self, "allowed_executables", tuple(allowed))
        # Force config through strict JSON so NaN cannot enter the hash.
        strict_load(canonical_bytes(_plain_config(self.config)))


def _reject_env(env: Mapping[str, str]) -> None:
    if not isinstance(env, Mapping):
        raise SpecError("invalid_spec", "env must be a mapping")
    for key, value in env.items():
        if not isinstance(key, str) or not key.isidentifier() or not isinstance(value, str) or "\x00" in value:
            raise SpecError("invalid_spec", "env entries must be identifier keys and string values")
        upper = key.upper()
        if any(marker in upper for marker in _SECRET) or upper in {"PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"}:
            raise SpecError("invalid_spec", "env contains a blocked variable")


def _plain_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise SpecError("invalid_spec", "config must be a mapping")
    return dict(config)


def spec_to_dict(spec: ExperimentSpec) -> dict[str, Any]:
    contract = None
    if spec.metric_contract is not None:
        contract = {
            "relative_path": spec.metric_contract.relative_path,
            "required_keys": list(spec.metric_contract.required_keys),
            "maxima": {key: spec.metric_contract.maxima[key] for key in sorted(spec.metric_contract.maxima)},
            "minima": {key: spec.metric_contract.minima[key] for key in sorted(spec.metric_contract.minima)},
            "timeout_counts_as_incorrect": spec.metric_contract.timeout_counts_as_incorrect,
        }
    return {
        "schema_version": 1,
        "run_id": spec.run_id,
        "task_id": spec.task_id,
        "workspace": str(spec.workspace),
        "steps": [{"name": step.name, "argv": list(step.argv), "cwd": step.cwd} for step in spec.steps],
        "limits": {
            "wall_time_s": spec.limits.wall_time_s,
            "cpu_time_s": spec.limits.cpu_time_s,
            "memory_mb": spec.limits.memory_mb,
            "max_output_bytes": spec.limits.max_output_bytes,
            "max_file_bytes": spec.limits.max_file_bytes,
            "max_child_processes": spec.limits.max_child_processes,
        },
        "source_files": list(spec.source_files),
        "protected_paths": [str(path) for path in spec.protected_paths],
        "metric_contract": contract,
        "config": _plain_config(spec.config),
        "env": {key: spec.env[key] for key in sorted(spec.env)},
        "backend": spec.backend,
        "isolation": spec.isolation,
        "network_policy": spec.network_policy,
        "docker_image": spec.docker_image,
        "allowed_executables": list(spec.allowed_executables),
        "mlx_min_available_mb": spec.mlx_min_available_mb,
    }


def spec_sha256(spec: ExperimentSpec) -> str:
    return sha256_json(spec_to_dict(spec))


def config_sha256(spec: ExperimentSpec) -> str:
    return sha256_json(_plain_config(spec.config))


def load_spec(payload: Mapping[str, Any] | str | bytes | Path) -> ExperimentSpec:
    if isinstance(payload, Path):
        document = strict_load(payload.read_bytes())
    elif isinstance(payload, (str, bytes)):
        document = strict_load(payload)
    else:
        document = strict_load(canonical_bytes(dict(payload)))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SpecError("invalid_spec", "spec schema_version must be 1")
    limits = document.get("limits") or {}
    if not isinstance(limits, dict):
        raise SpecError("invalid_spec", "limits must be an object")
    steps_raw = document.get("steps")
    if not isinstance(steps_raw, list):
        raise SpecError("invalid_spec", "steps must be a list")
    steps = tuple(
        CommandStep(name=item["name"], argv=tuple(item["argv"]), cwd=item.get("cwd", "."))
        for item in steps_raw
    )
    contract_raw = document.get("metric_contract")
    contract = None
    if contract_raw is not None:
        if not isinstance(contract_raw, dict):
            raise SpecError("invalid_spec", "metric_contract must be an object")
        contract = MetricContract(
            relative_path=contract_raw["relative_path"],
            required_keys=tuple(contract_raw["required_keys"]),
            maxima=dict(contract_raw.get("maxima") or {}),
            minima=dict(contract_raw.get("minima") or {}),
            timeout_counts_as_incorrect=contract_raw.get("timeout_counts_as_incorrect", False),
        )
    return ExperimentSpec(
        run_id=document["run_id"],
        task_id=document["task_id"],
        workspace=Path(document["workspace"]),
        steps=steps,
        limits=RunLimits(
            wall_time_s=limits.get("wall_time_s", 30),
            cpu_time_s=limits.get("cpu_time_s", 30),
            memory_mb=limits.get("memory_mb", 256),
            max_output_bytes=limits.get("max_output_bytes", 100_000),
            max_file_bytes=limits.get("max_file_bytes", 8_388_608),
            max_child_processes=limits.get("max_child_processes"),
        ),
        source_files=tuple(document.get("source_files") or ()),
        protected_paths=tuple(Path(item) for item in document.get("protected_paths") or ()),
        metric_contract=contract,
        config=dict(document.get("config") or {}),
        env=dict(document.get("env") or {}),
        backend=document.get("backend", "cpu"),
        isolation=document.get("isolation", "process_group"),
        network_policy=document.get("network_policy", "deny"),
        docker_image=document.get("docker_image"),
        allowed_executables=tuple(document.get("allowed_executables") or ()),
        mlx_min_available_mb=document.get("mlx_min_available_mb", 2048),
    )
