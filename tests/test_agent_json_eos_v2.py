from __future__ import annotations

import numpy as np
import pytest

from vare.agent_json_eos_v2 import CHAT_START_MARKER, make_json_object_eos_v2_logits_processor
from vare.agent_json_eos_v3 import make_json_object_eos_v3_logits_processor


class _Tokenizer:
    eos_token_id = 0
    unk_token_id = 99
    _text = {
        0: "",
        1: '{"name":"finish","arguments":{}}',
        3: CHAT_START_MARKER,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == CHAT_START_MARKER
        return 3

    def decode(self, tokens, skip_special_tokens=False) -> str:
        return "".join(self._text[int(token)] for token in tokens)


class _Mx:
    array = staticmethod(np.array)
    arange = staticmethod(np.arange)
    any = staticmethod(np.any)
    where = staticmethod(np.where)


def test_v2_blocks_chat_start_at_generation_boundary_without_rewriting_output() -> None:
    tokenizer = _Tokenizer()
    processor, state = make_json_object_eos_v2_logits_processor(
        tokenizer, prompt_token_count=2, eos_ids={0}, mx=_Mx,
    )
    initial = np.zeros((1, 5), dtype=np.float32)

    blocked = processor(np.array([8, 9, 3]), initial)

    assert np.isneginf(blocked[0, 3])
    assert np.all(blocked[0, [0, 1, 2, 4]] == 0)
    assert tokenizer.decode([3], skip_special_tokens=False) == CHAT_START_MARKER
    assert state["chat_start_marker_blocked"] is True


def test_v2_forces_eos_after_complete_json_and_leaves_parser_in_charge() -> None:
    tokenizer = _Tokenizer()
    processor, state = make_json_object_eos_v2_logits_processor(
        tokenizer, prompt_token_count=1, eos_ids={0}, mx=_Mx,
    )
    logits = np.zeros((1, 5), dtype=np.float32)

    stopped = processor(np.array([8, 1]), logits)

    assert np.isfinite(stopped[0, 0])
    assert np.all(np.isneginf(stopped[0, 1:]))
    assert state["completion_seen"] is True


def test_v2_rejects_unresolvable_or_mismatched_chat_marker_token() -> None:
    class BadTokenizer(_Tokenizer):
        def decode(self, tokens, skip_special_tokens=False) -> str:
            return "wrong" if int(tokens[0]) == 3 else super().decode(tokens, skip_special_tokens)

    with pytest.raises(ValueError, match="does not decode"):
        make_json_object_eos_v2_logits_processor(
            BadTokenizer(), prompt_token_count=0, eos_ids={0}, mx=_Mx,
        )


def test_v3_blocks_premature_eos_as_well_as_chat_marker() -> None:
    tokenizer = _Tokenizer()
    processor, state = make_json_object_eos_v3_logits_processor(
        tokenizer, prompt_token_count=2, eos_ids={0}, mx=_Mx,
    )
    logits = np.zeros((1, 5), dtype=np.float32)

    constrained = processor(np.array([8, 9]), logits)

    assert np.isneginf(constrained[0, 0])
    assert np.isneginf(constrained[0, 3])
    assert np.all(constrained[0, [1, 2, 4]] == 0)
    assert state["premature_eos_blocked"] is True


def test_v3_releases_only_eos_after_complete_json() -> None:
    tokenizer = _Tokenizer()
    processor, state = make_json_object_eos_v3_logits_processor(
        tokenizer, prompt_token_count=1, eos_ids={0}, mx=_Mx,
    )
    logits = np.zeros((1, 5), dtype=np.float32)

    constrained = processor(np.array([9, 1]), logits)

    assert np.isfinite(constrained[0, 0])
    assert np.all(np.isneginf(constrained[0, 1:]))
    assert state["completion_seen"] is True
