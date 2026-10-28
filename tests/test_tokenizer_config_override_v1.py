import hashlib
import json

import pytest

from scripts.run_local_agent import tokenizer_override


def test_override_is_isolated_and_binds_both_configurations(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    source = model / "tokenizer_config.json"
    source.write_text(json.dumps({"chat_template": "original", "eos_token": "end"}))
    before = source.read_bytes()
    override = tmp_path / "corrected.json"
    override.write_text(json.dumps({"chat_template": "corrected", "eos_token": "end"}))
    arguments, identity = tokenizer_override(model, override)
    assert arguments == {"chat_template": "corrected"}
    assert identity["original_config_sha256"] == hashlib.sha256(before).hexdigest()
    assert identity["override_config_sha256"] == hashlib.sha256(override.read_bytes()).hexdigest()
    assert source.read_bytes() == before


@pytest.mark.parametrize("changed", [{"chat_template": "new", "eos_token": "changed"},
                                     {"chat_template": ""}, {"chat_template": 7}])
def test_override_cannot_change_tokenizer_semantics_beyond_template(tmp_path, changed):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "old", "eos_token": "end"}))
    override = tmp_path / "override.json"
    override.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="only chat_template"):
        tokenizer_override(tmp_path, override)


def test_default_path_binds_the_original_template_without_modifying_it(tmp_path):
    original = tmp_path / "tokenizer_config.json"
    original.write_text(json.dumps({"chat_template": "frozen", "eos_token": "end"}))
    before = original.read_bytes()

    arguments, identity = tokenizer_override(tmp_path, None)

    assert arguments == {}
    assert identity["original_config_sha256"] == hashlib.sha256(before).hexdigest()
    assert identity["effective_chat_template_sha256"] == hashlib.sha256(b"frozen").hexdigest()
    assert identity["effective_template_source"] == "snapshot"
    assert original.read_bytes() == before
