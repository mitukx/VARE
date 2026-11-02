from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import scripts.run_local_agent as agent


class _Tokenizer:
    bos_token = "<bos>"

    def __init__(self):
        self.special_token_flags = []

    def apply_chat_template(self, _messages, **_kwargs):
        return "<bos>rendered-prompt"

    def encode(self, _text, *, add_special_tokens):
        self.special_token_flags.append(add_special_tokens)
        return [21, 22] if not add_special_tokens else [20, 21, 22]


class _LogprobVector:
    def __getitem__(self, token):
        return SimpleNamespace(item=lambda: -0.25 * token)


def test_prompt_id_helper_matches_mlx_lm_bos_rule():
    tokenizer = _Tokenizer()
    assert agent.mlx_stream_prompt_ids(tokenizer, "<bos>body") == [21, 22]
    assert tokenizer.special_token_flags[-1] is False
    assert agent.mlx_stream_prompt_ids(tokenizer, "body") == [20, 21, 22]
    assert tokenizer.special_token_flags[-1] is True


def test_mlx_agent_trace_records_exact_tokens_and_selected_behavior_logprobs(
    monkeypatch,
):
    calls = {}
    response_stream = (
        SimpleNamespace(
            token=41, text="hello", logprobs=_LogprobVector(), finish_reason=None
        ),
        SimpleNamespace(
            token=42, text="", logprobs=_LogprobVector(), finish_reason="stop"
        ),
    )

    def make_sampler(**kwargs):
        calls["sampler"] = kwargs
        return "full-softmax-sampler"

    def stream_generate(model, tokenizer, *, prompt, max_tokens, sampler):
        calls["generation"] = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        return iter(response_stream)

    mlx_lm = ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate
    sample_utils = ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)

    tokenizer = _Tokenizer()
    args = SimpleNamespace(
        max_context_tokens=32,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        trace_on_policy=True,
    )
    raw, completion_count, prompt_count, limit = agent.generate_turn(
        "mlx",
        object(),
        tokenizer,
        [],
        [],
        limit=8,
        args=args,
    )

    assert raw == "hello"
    assert (completion_count, prompt_count, limit) == (2, 2, 8)
    assert calls["generation"]["prompt"] == [21, 22]
    assert calls["sampler"] == {"temp": 1.0, "top_p": 1.0, "min_p": 0.0, "top_k": 0}
    trace = args._trace_last_generation
    assert trace["prompt_token_ids"] == [21, 22]
    assert trace["completion_token_ids"] == [41, 42]
    assert trace["behavior_logprobs"] == [-10.25, -10.5]
    assert trace["behavior_logprob_count"] == len(trace["completion_token_ids"])
    assert trace["sampler"]["distribution"] == "untruncated_model_softmax"
    assert trace["finish_reason"] == "stop"
    assert trace["prompt_token_ids_sha256"] == hashlib.sha256(b"[21,22]").hexdigest()
    assert (
        trace["completion_token_ids_sha256"] == hashlib.sha256(b"[41,42]").hexdigest()
    )
    assert (
        trace["behavior_logprobs_sha256"]
        == hashlib.sha256(b"[-10.25,-10.5]").hexdigest()
    )


def test_trace_mode_fails_closed_on_non_exact_sampling_configuration():
    tokenizer = _Tokenizer()
    args = SimpleNamespace(
        max_context_tokens=32,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        trace_on_policy=True,
    )
    try:
        agent.generate_turn("mlx", object(), tokenizer, [], [], limit=8, args=args)
    except ValueError as exc:
        assert "temperature=1" in str(exc)
    else:
        raise AssertionError("on-policy trace accepted truncated sampling parameters")
