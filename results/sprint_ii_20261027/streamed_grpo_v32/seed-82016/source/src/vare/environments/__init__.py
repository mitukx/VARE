"""Executable repository environments for compute-efficient agent/RL research."""

from .campaign import CampaignRecord, CampaignSummary, CommandWorkspaceAgent, EnvironmentCampaignRunner
from .catalog import CatalogEnvironmentVerifier, FailureMatchedEnvironmentGenerator, TaskCatalog, WorkspaceAgent, WorkspaceAgentRollout
from .factory import GitTaskTemplate, freeze_protected_paths, resolve_revision
from .runner import CommandResult, EnvironmentResult, ExecutableEvaluator, MetricResult, Workspace, metric_map
from .spec import CommandSpec, EngineeringTaskSpec, MetricSpec, ProtectedPath, RepositorySource, ResourceLimits
from .verifier import ExecutableEnvironmentVerifier

__all__ = [
    "CampaignRecord",
    "CampaignSummary",
    "CommandWorkspaceAgent",
    "EnvironmentCampaignRunner",
    "CatalogEnvironmentVerifier",
    "FailureMatchedEnvironmentGenerator",
    "TaskCatalog",
    "WorkspaceAgent",
    "WorkspaceAgentRollout",
    "CommandResult",
    "CommandSpec",
    "EngineeringTaskSpec",
    "EnvironmentResult",
    "ExecutableEnvironmentVerifier",
    "ExecutableEvaluator",
    "GitTaskTemplate",
    "MetricResult",
    "MetricSpec",
    "ProtectedPath",
    "RepositorySource",
    "ResourceLimits",
    "Workspace",
    "freeze_protected_paths",
    "metric_map",
    "resolve_revision",
]
