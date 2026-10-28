import json
from ast import literal_eval

import pytest

from scripts.repair_qwen25_tool_template_v1 import BAD_EXAMPLE, repair


def render_example(template):
    # This fixture's Jinja expression is only a quoted string literal. The
    # actual cached checkpoint is separately rendered in the retained audit.
    rendered = literal_eval(template[2:-2].strip())
    return rendered.replace("<function-name>", '"search_source"').replace(
        "<args-json-object>", '{"query":"needle"}',
    )


def test_repair_changes_literal_example_to_valid_json():
    config = {"chat_template": '{{ "' + BAD_EXAMPLE + '" }}', "eos_token": "<eos>"}
    with pytest.raises(json.JSONDecodeError):
        json.loads(render_example(config["chat_template"]))
    fixed = repair(config)
    assert json.loads(render_example(fixed["chat_template"])) == {
        "name": "search_source", "arguments": {"query": "needle"},
    }
    assert fixed["eos_token"] == config["eos_token"]
    assert BAD_EXAMPLE in config["chat_template"]


@pytest.mark.parametrize("value", [None, "ordinary template", BAD_EXAMPLE * 2])
def test_unrecognized_or_ambiguous_input_is_rejected(value):
    with pytest.raises(ValueError):
        repair({"chat_template": value})
