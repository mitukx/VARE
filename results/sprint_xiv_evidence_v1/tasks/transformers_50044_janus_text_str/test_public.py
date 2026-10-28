import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers.models.janus.processing_janus import DEFAULT_SYSTEM_PROMPT, JanusProcessor
from transformers.processing_utils import ProcessorMixin


def _forwarded(text, *, mode, default_prompt=False):
    processor = JanusProcessor.__new__(JanusProcessor)
    processor.tokenizer = SimpleNamespace(init_kwargs={})
    processor.use_default_system_prompt = default_prompt
    processor.image_start_token = "<public-image-start>"
    processor._merge_kwargs = lambda *_args, **_kwargs: {"text_kwargs": {"generation_mode": mode}}

    previous = ProcessorMixin.__call__
    ProcessorMixin.__call__ = lambda _self, *, images=None, text=None, **_kwargs: text
    try:
        return processor(text, text_kwargs={"generation_mode": mode})
    finally:
        ProcessorMixin.__call__ = previous


def test_single_string_is_one_item_in_text_mode_with_default_prompt():
    assert _forwarded("cedar", mode="text", default_prompt=True) == [DEFAULT_SYSTEM_PROMPT + "cedar"]


def test_single_string_is_one_item_in_image_mode():
    assert _forwarded("cedar", mode="image") == ["cedar<public-image-start>"]


def test_list_input_keeps_per_prompt_behavior():
    assert _forwarded(["cedar", "maple"], mode="image") == [
        "cedar<public-image-start>",
        "maple<public-image-start>",
    ]


if __name__ == "__main__":
    test_single_string_is_one_item_in_text_mode_with_default_prompt()
    test_single_string_is_one_item_in_image_mode()
    test_list_input_keeps_per_prompt_behavior()
