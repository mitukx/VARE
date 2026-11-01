from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from ..evidence import sha256_file
from .spec import (
    CommandSpec,
    EngineeringTaskSpec,
    MetricSpec,
    ProtectedPath,
    RepositorySource,
    ResourceLimits,
)


@dataclass(frozen=True, slots=True)
class GitTaskTemplate:
    id: str
    title: str
    prompt: str
    family: str
    clone_url: str
    base_revision: str
    tests: tuple[CommandSpec, ...]
    metrics: tuple[MetricSpec, ...] = ()
    limits: ResourceLimits = ResourceLimits()
    correctness_weight: float = 0.8

    def build(self) -> EngineeringTaskSpec:
        return EngineeringTaskSpec(
            id=self.id,
            title=self.title,
            prompt=self.prompt,
            family=self.family,
            source=RepositorySource(clone_url=self.clone_url, base_revision=self.base_revision),
            tests=self.tests,
            metrics=self.metrics,
            limits=self.limits,
            correctness_weight=self.correctness_weight,
        )


def freeze_protected_paths(spec: EngineeringTaskSpec, workspace: str | Path, paths: list[str]) -> EngineeringTaskSpec:
    root = Path(workspace).resolve()
    protected: list[ProtectedPath] = []
    for rel in paths:
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"path escapes workspace: {rel}")
        if not path.is_file():
            raise FileNotFoundError(path)
        protected.append(ProtectedPath(path=rel, sha256=sha256_file(path)))
    return EngineeringTaskSpec(
        id=spec.id,
        title=spec.title,
        prompt=spec.prompt,
        family=spec.family,
        source=spec.source,
        tests=spec.tests,
        metrics=spec.metrics,
        protected_paths=tuple(protected),
        limits=spec.limits,
        correctness_weight=spec.correctness_weight,
        metadata=dict(spec.metadata),
    )


def resolve_revision(repo: str | Path, revision: str) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=Path(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"cannot resolve revision {revision}: {proc.stderr.strip()}")
    return proc.stdout.strip()
