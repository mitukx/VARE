"""Public regression for structured outputs during device movement."""

import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.pytorch_utils import move_to_device


def test_move_to_device_preserves_model_output_structure():
    value = BaseModelOutput(
        last_hidden_state=torch.zeros(1, 2, 4),
        hidden_states=(torch.zeros(2, 4), torch.ones(2, 4)),
    )
    moved = move_to_device(value, torch.device("cpu"))
    assert isinstance(moved, BaseModelOutput)
    assert isinstance(moved.hidden_states, tuple)
