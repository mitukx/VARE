"""Public regression for repeated-window accounting."""

import torch
from types import SimpleNamespace
from transformers import WatermarkDetector


def test_detector_counts_each_repeated_ngram_once():
    detector = WatermarkDetector(
        SimpleNamespace(vocab_size=128, bos_token_id=127, is_encoder_decoder=False, decoder_start_token_id=None),
        "cpu",
        {"greenlist_ratio": 0.25, "bias": 2.0, "seeding_scheme": "selfhash", "context_width": 1},
        ignore_repeated_ngrams=True,
    )
    result = detector(torch.tensor([[3] * 10], dtype=torch.long), return_dict=True)
    assert result.num_tokens_scored.tolist() == [1.0]
    assert 0.0 <= result.num_green_tokens[0] <= 1.0
