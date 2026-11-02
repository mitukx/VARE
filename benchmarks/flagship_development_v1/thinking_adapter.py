"""Candidate-side invocation only; expected outputs remain in the host."""
import importlib.util
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

spec = importlib.util.spec_from_file_location("candidate_tokenizer", Path(__file__).with_name("target.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def run(request):
    vocab = {token: i for i, token in enumerate(["<unk>", *request["tokens"]])}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.chat_template = request["template"]
    wrapped = candidate.TokenizerWrapper(tokenizer)
    return {
        "start": wrapped.think_start, "end": wrapped.think_end,
        "start_tokens": wrapped.think_start_tokens,
        "end_tokens": wrapped.think_end_tokens,
        "rendered": wrapped.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=False,
            **request.get("kwargs", {}),
        ),
    }
