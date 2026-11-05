"""ResearchGym experiment execution.

This package runs bounded experiments and recovers committed results after
interruption. It does not own task definitions, graders, or the operator CLI.
"""
from .headroom import HeadroomDecision, decide_accelerator_headroom
from .isolation import RESIDUAL_RISK
from .runner import ExperimentRunner, RunView
from .spec import CommandStep, ExperimentSpec, MetricContract, RunLimits, load_spec

__all__ = [
    "CommandStep",
    "ExperimentRunner",
    "ExperimentSpec",
    "HeadroomDecision",
    "MetricContract",
    "RESIDUAL_RISK",
    "RunLimits",
    "RunView",
    "decide_accelerator_headroom",
    "load_spec",
]
