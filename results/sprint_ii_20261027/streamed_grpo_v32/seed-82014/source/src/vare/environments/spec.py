from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


MetricDirection = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    wall_time_s: float = 30.0
    cpu_time_s: int | None = 30
    memory_mb: int | None = 2048
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.wall_time_s <= 0:
            raise ValueError("wall_time_s must be positive")
        if self.cpu_time_s is not None and self.cpu_time_s <= 0:
            raise ValueError("cpu_time_s must be positive when set")
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive when set")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")


@dataclass(frozen=True, slots=True)
class CommandSpec:
    argv: tuple[str, ...]
    name: str = "command"
    cwd: str = "."
    cwd_base: Literal["workspace", "evaluator"] = "workspace"
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(x, str) and x for x in self.argv):
            raise ValueError("argv must contain at least one non-empty string")
        if Path(self.cwd).is_absolute():
            raise ValueError("command cwd must be relative to its declared base")
        if self.cwd_base not in {"workspace", "evaluator"}:
            raise ValueError("cwd_base must be workspace or evaluator")


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    command: CommandSpec
    json_key: str
    direction: MetricDirection = "min"
    weight: float = 1.0
    max_regression_fraction: float | None = 0.05

    def __post_init__(self) -> None:
        if not self.name or not self.json_key:
            raise ValueError("metric name/json_key must be non-empty")
        if self.direction not in {"min", "max"}:
            raise ValueError("metric direction must be min or max")
        if self.weight < 0:
            raise ValueError("metric weight must be non-negative")
        if self.max_regression_fraction is not None and self.max_regression_fraction < 0:
            raise ValueError("max_regression_fraction must be non-negative")


@dataclass(frozen=True, slots=True)
class ProtectedPath:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if Path(self.path).is_absolute():
            raise ValueError("protected path must be workspace-relative")
        if len(self.sha256) != 64:
            raise ValueError("protected path requires a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RepositorySource:
    local_path: str | None = None
    clone_url: str | None = None
    base_revision: str | None = None

    def __post_init__(self) -> None:
        if bool(self.local_path) == bool(self.clone_url):
            raise ValueError("exactly one of local_path or clone_url is required")


@dataclass(frozen=True, slots=True)
class EngineeringTaskSpec:
    id: str
    title: str
    prompt: str
    family: str
    source: RepositorySource
    tests: tuple[CommandSpec, ...]
    metrics: tuple[MetricSpec, ...] = ()
    protected_paths: tuple[ProtectedPath, ...] = ()
    limits: ResourceLimits = ResourceLimits()
    correctness_weight: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    _source_base: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id or not self.title or not self.prompt or not self.family:
            raise ValueError("task identity/title/prompt/family must be non-empty")
        if not self.tests:
            raise ValueError("at least one executable test command is required")
        if not 0.0 <= self.correctness_weight <= 1.0:
            raise ValueError("correctness_weight must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        # Loader context is deliberately transient: task manifests stay portable
        # and their content hashes do not depend on the local checkout path.
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "family": self.family,
            "source": asdict(self.source),
            "tests": [asdict(x) for x in self.tests],
            "metrics": [asdict(x) for x in self.metrics],
            "protected_paths": [asdict(x) for x in self.protected_paths],
            "limits": asdict(self.limits),
            "correctness_weight": self.correctness_weight,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, source_base: str | Path | None = None
    ) -> "EngineeringTaskSpec":
        source = RepositorySource(**raw["source"])
        tests = tuple(CommandSpec(**x) for x in raw["tests"])
        metrics = tuple(
            MetricSpec(
                name=x["name"],
                command=CommandSpec(**x["command"]),
                json_key=x["json_key"],
                direction=x.get("direction", "min"),
                weight=float(x.get("weight", 1.0)),
                max_regression_fraction=x.get("max_regression_fraction", 0.05),
            )
            for x in raw.get("metrics", [])
        )
        protected = tuple(ProtectedPath(**x) for x in raw.get("protected_paths", []))
        limits = ResourceLimits(**raw.get("limits", {}))
        return cls(
            id=raw["id"],
            title=raw["title"],
            prompt=raw["prompt"],
            family=raw["family"],
            source=source,
            tests=tests,
            metrics=metrics,
            protected_paths=protected,
            limits=limits,
            correctness_weight=float(raw.get("correctness_weight", 0.8)),
            metadata=dict(raw.get("metadata", {})),
            _source_base=None if source_base is None else str(Path(source_base).resolve()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EngineeringTaskSpec":
        p = Path(path).resolve()
        return cls.from_dict(
            json.loads(p.read_text(encoding="utf-8")), source_base=p.parent
        )

    def write(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
