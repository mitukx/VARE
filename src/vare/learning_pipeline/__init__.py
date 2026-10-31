"""Coding-policy learning pipeline v1 (Qwen3-4B, MLX LoRA, Apple Silicon).

Stages: curriculum build -> calibration -> base rollouts -> SFT / RFT LoRA
training -> held-out evaluation under identical budgets -> paired comparison.

Nothing in this package is evidence of model capability by itself. Runs with
``--mock`` or a tiny random model are synthetic plumbing checks and every
record they write carries ``"synthetic": true``.
"""
