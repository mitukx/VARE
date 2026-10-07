"""VARE: Verification-Aware Reinforcement Engine."""

from .config import EngineConfig, LagConfig, PromotionConfig, ReplayConfig
from .engine import CapabilityLoop, LoopHooks
from .types import Attempt, EvaluationReport, Task, Verification

__all__ = [
    "Attempt",
    "CapabilityLoop",
    "EngineConfig",
    "EvaluationReport",
    "LagConfig",
    "LoopHooks",
    "PromotionConfig",
    "ReplayConfig",
    "Task",
    "Verification",
]

__version__ = "0.3.1"
