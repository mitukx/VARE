"""Small GRPO trainer for integer expressions.

The loss math does not import MLX or PyTorch. Model backends live in
``mlx_train`` and ``torch_train`` and are imported only when a run starts.
"""

from minigrpo.grpo import group_advantages, sequence_loss

__all__ = ["group_advantages", "sequence_loss"]
