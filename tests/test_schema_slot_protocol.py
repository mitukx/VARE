import json

import pytest

from scripts.run_local_agent import (
    controller_source_hashes,
    parse_tool_output_v1,
    replace_text_slot_guidance,
    tools_for,
)
from vare.schema_slot_protocol import (
    SchemaSlotError,
    allowed_choice_tokens,
    generation_protocol_failure,
    terminal_newline_is_present,
    serialize_slot_action,
    tool_choice_token_sequences,
)


class _ToyTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=False):
        return ''.join(chr(token) for token in tokens)


def test_choice_prefix_only_permits_exposed_tool_names_and_eos_at_completion():
    tokenizer = _ToyTokenizer()
    tools = tools_for('structured_replace', public_tests=True,
                      tool_interface='lease_bound_navigation_v1')
    sequences = tool_choice_token_sequences(tokenizer, tools)
    assert allowed_choice_tokens([], sequences, {tokenizer.eos_token_id}) == {
        name.encode()[0] for name in sequences
    }
    assert allowed_choice_tokens(list(b'list_file'), sequences, {99}) == {ord('s')}
    assert allowed_choice_tokens(list(b'list_files'), sequences, {99}) == {99}
    assert allowed_choice_tokens(list(b'list_files') + [99], sequences, {99}) is None
    with pytest.raises(ValueError, match='outside the allowed vocabulary'):
        allowed_choice_tokens(list(b'not-a-tool'), sequences, {99})


def test_schema_slot_contract_covers_every_navigation_tool_without_extra_keys():
    tools = tools_for('structured_replace', public_tests=True,
                      tool_interface='lease_bound_navigation_v1')
    string_values = {
        'path': 'src/model.py', 'query': 'cache_key', 'lease': 'lease-v1',
        'text': 'return value + 1\n', 'summary': 'Fixed the cache key.',
    }
    integer_values = {'start': '4', 'end': '18'}
    expected_names = {
        'list_files', 'search_source', 'read_lines', 'replace_lines',
        'check_syntax', 'run_public_tests', 'finish',
    }
    assert {tool['function']['name'] for tool in tools} == expected_names

    for tool in tools:
        name = tool['function']['name']
        schema = tool['function']['parameters']
        raw = {}
        for key in schema.get('required', []):
            if schema['properties'][key]['type'] == 'integer':
                raw[key] = integer_values[key]
            else:
                raw[key] = string_values[key]
        encoded, payload = serialize_slot_action(name, raw, tools)
        status, parsed = parse_tool_output_v1(encoded, tools)
        assert status == 'complete_tool_call'
        assert parsed['name'] == name
        assert parsed['arguments'] == payload['arguments']
        assert set(payload['arguments']) == set(schema.get('required', []))


@pytest.mark.parametrize('raw', ['4.0', 'true', '"4"', '04', '4 trailing'])
def test_integer_slots_fail_closed_without_coercion(raw):
    tools = tools_for('structured_replace', tool_interface='lease_bound_navigation_v1')
    with pytest.raises(SchemaSlotError):
        serialize_slot_action('read_lines', {'path': 'src/model.py', 'start': raw, 'end': '5'}, tools)


def test_string_slot_preserves_exact_raw_model_value():
    tools = tools_for('structured_replace', tool_interface='lease_bound_navigation_v1')
    exact = ' first line\nsecond line  '
    encoded, payload = serialize_slot_action('replace_lines', {'lease': 'L1', 'text': exact}, tools)
    assert payload['arguments']['text'] == exact
    assert json.loads(encoded)['arguments']['text'] == exact


def test_terminal_newline_constraint_checks_exact_generated_string_suffix():
    tokenizer = _ToyTokenizer()
    assert not terminal_newline_is_present(list(b'return a + b'), tokenizer)
    assert terminal_newline_is_present(list(b'return a + b\n'), tokenizer)
    assert not terminal_newline_is_present([], tokenizer)


@pytest.mark.parametrize('protocol', ['vare.schema-slots.v1', 'vare.schema-slots.v2'])
def test_internal_slot_budget_failure_is_not_a_parser_failure(protocol):
    detail = {'protocol_id': protocol, 'stop_reason': 'argument_slot_token_budget_exhausted'}
    assert generation_protocol_failure(detail) == 'argument_slot_token_budget_exhausted'


def test_unknown_protocol_failure_is_not_consumed_as_internal():
    assert generation_protocol_failure({'protocol_id': 'other', 'stop_reason': 'argument_slot_token_budget_exhausted'}) is None


def test_schema_slot_controller_source_is_in_run_identity():
    sources = controller_source_hashes('lease_bound_navigation_v1', 'schema_slots_v2')
    assert sources['src/vare/schema_slot_protocol.py']
    assert sources['scripts/run_local_agent.py']


def test_source_only_edit_guidance_is_versioned_and_explicit():
    assert replace_text_slot_guidance('original') == ''
    guidance = replace_text_slot_guidance('source_only_v1')
    assert 'smallest complete replacement as plain source lines' in guidance
    assert 'Do not use Markdown fences' in guidance
    assert 'tool-call syntax' in guidance
    assert 'explanatory prose' in guidance
    with pytest.raises(ValueError, match='unknown replace-text guidance'):
        replace_text_slot_guidance('v99')
