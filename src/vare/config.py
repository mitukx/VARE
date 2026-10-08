from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LagConfig:
    max_policy_lag: int = 2
    max_verifier_lag: int = 2
    max_shift: float = 0.35
    policy_decay: float = 0.35
    verifier_decay: float = 0.45
    shift_decay: float = 1.25


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 50_000
    min_priority: float = 1e-4
    failure_bonus: float = 1.5
    disagreement_bonus: float = 1.0
    freshness_floor: float = 0.05


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    min_primary_gain: float = 0.01
    max_slice_regression: float = 0.03
    max_relative_cost_increase: float = 0.20
    max_verifier_disagreement: float = 0.20
    min_eval_examples: int = 64
    paired_confidence_gate: bool = False
    min_paired_examples: int = 64
    paired_alpha: float = 0.05
    paired_bootstrap_samples: int = 2000
    # Opt-in strict post-training gates; legacy/demo defaults remain compatible.
    require_complete_slices: bool = False
    require_measured_disagreement: bool = False
    require_reward_audit: bool = False
    reward_audit_max_false_accept_ucb: float = 0.25
    reward_audit_min_proxy_positives: int = 32
    reward_audit_alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class EngineConfig:
    rollout_concurrency: int = 16
    replay_batch_size: int = 256
    samples_per_task: int = 1
    preserve_rollout_groups: bool = True
    generated_tasks_per_round: int = 32
    generated_task_buffer: int = 2048
    lag: LagConfig = field(default_factory=LagConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
