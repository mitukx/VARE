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
    allowed_prefix_tokens,
    generation_protocol_failure,
    function_body_normalization_choices,
    dynamic_function_slot_choices,
    single_line_termination_state,
    single_line_token_piece_is_safe,
    prefix_token_sequences,
    string_choice_token_sequences,
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


def test_v7_array_slot_round_trips_only_strict_json_string_arrays():
    tools = tools_for('structured_replace', tool_interface='lease_bound_function_edit_v2')
    tool = next(item for item in tools
                if item['function']['name'] == 'replace_function_body_transactional')
    schema = tool['function']['parameters']
    assert schema['additionalProperties'] is False
    assert schema['properties']['body_lines']['type'] == 'array'
    raw = {'lease': 'lease-v1', 'body_lines': '["    return max(0, value)", ""]'}

    encoded, payload = serialize_slot_action(
        'replace_function_body_transactional', raw, tools,
    )

    assert payload['arguments'] == {
        'lease': 'lease-v1', 'body_lines': ['    return max(0, value)', ''],
    }
    status, parsed = parse_tool_output_v1(encoded, tools)
    assert status == 'complete_tool_call'
    assert parsed['name'] == payload['name']
    assert parsed['arguments'] == payload['arguments']


def test_v8_function_navigation_and_edit_values_are_bound_to_prior_tool_results():
    messages = [
        {'role': 'tool', 'name': 'list_functions', 'content': json.dumps({
            'status': 'functions_listed', 'path': 'module.py',
            'functions': [{'qualified_name': 'clamp'}, {'qualified_name': 'clamp'}],
        })},
        {'role': 'tool', 'name': 'read_function', 'content': json.dumps({
            'status': 'read_complete', 'path': 'module.py', 'lease': 'lease-raw-7',
        })},
    ]
    assert dynamic_function_slot_choices('read_function', 'path', messages) == (
        ['module.py'], 'prior_list_functions_path_v1',
    )
    assert dynamic_function_slot_choices('read_function', 'qualified_name', messages) == (
        ['clamp'], 'prior_list_functions_names_v1',
    )
    assert dynamic_function_slot_choices(
        'replace_function_body_transactional', 'lease', messages,
    ) == (['lease-raw-7'], 'prior_read_function_lease_v1')
    assert dynamic_function_slot_choices('read_lines', 'path', messages) is None


def test_v8_dynamic_function_values_fail_closed_without_successful_provenance():
    missing = dynamic_function_slot_choices('read_function', 'qualified_name', [])
    malformed = dynamic_function_slot_choices('read_function', 'path', [
        {'role': 'tool', 'name': 'list_functions', 'content': '{bad json'},
    ])
    assert missing == ([], 'dynamic_choice_source_missing:list_functions')
    assert malformed == ([], 'dynamic_choice_source_invalid:list_functions')


def test_v9_exact_text_edit_lease_is_bound_to_a_successful_read_lines_result():
    messages = [{
        'role': 'tool', 'name': 'read_lines', 'content': json.dumps({
            'status': 'read_complete', 'path': 'module.py', 'lease': 'line-lease-9',
        }),
    }]
    assert dynamic_function_slot_choices(
        'replace_exact_text_transactional', 'lease', messages,
        include_transactional_exact_text=True,
    ) == (['line-lease-9'], 'prior_read_lines_lease_v1')
    assert dynamic_function_slot_choices(
        'replace_exact_text_transactional', 'lease', messages,
    ) is None
    assert dynamic_function_slot_choices(
        'replace_exact_text_transactional', 'lease', [],
        include_transactional_exact_text=True,
    ) == ([], 'dynamic_choice_source_missing:read_lines')


@pytest.mark.parametrize('raw_array', [
    '    return max(0, value)',
    '["    return max(0, value)"] trailing',
    '["line 1", 2]',
    '[]',
])
def test_v7_array_slot_rejects_non_json_or_non_string_or_empty_payloads(raw_array):
    tools = tools_for('structured_replace', tool_interface='lease_bound_function_edit_v2')
    with pytest.raises(SchemaSlotError):
        serialize_slot_action(
            'replace_function_body_transactional',
            {'lease': 'lease-v1', 'body_lines': raw_array}, tools,
        )


def test_terminal_newline_constraint_checks_exact_generated_string_suffix():
    tokenizer = _ToyTokenizer()
    assert not terminal_newline_is_present(list(b'return a + b'), tokenizer)
    assert terminal_newline_is_present(list(b'return a + b\n'), tokenizer)
    assert not terminal_newline_is_present([], tokenizer)


@pytest.mark.parametrize('protocol', ['vare.schema-slots.v1', 'vare.schema-slots.v2', 'vare.schema-slots.v3', 'vare.schema-slots.v4', 'vare.schema-slots.v5', 'vare.schema-slots.v6', 'vare.schema-slots.v7', 'vare.schema-slots.v8', 'vare.schema-slots.v9', 'vare.schema-slots.v10'])
def test_internal_slot_budget_failure_is_not_a_parser_failure(protocol):
    detail = {'protocol_id': protocol, 'stop_reason': 'argument_slot_token_budget_exhausted'}
    assert generation_protocol_failure(detail) == 'argument_slot_token_budget_exhausted'


def test_unknown_protocol_failure_is_not_consumed_as_internal():
    assert generation_protocol_failure({'protocol_id': 'other', 'stop_reason': 'argument_slot_token_budget_exhausted'}) is None


def test_schema_slot_controller_source_is_in_run_identity():
    sources = controller_source_hashes('lease_bound_navigation_v1', 'schema_slots_v2')
    assert sources['src/vare/schema_slot_protocol.py']
    assert sources['scripts/run_local_agent.py']

    exact_text = controller_source_hashes(
        'lease_bound_exact_text_transactional_v1', 'schema_slots_v9',
    )
    assert exact_text['src/vare/schema_slot_protocol.py']
    assert exact_text['src/vare/source_navigation_v2.py']

    cached_slots = controller_source_hashes(
        'lease_bound_exact_text_transactional_v1', 'schema_slots_v10',
    )
    assert cached_slots['src/vare/schema_slot_protocol.py']
    assert cached_slots['scripts/run_local_agent.py']


@pytest.mark.parametrize(('raw', 'original_newline', 'expected'), [
    ('    return value\n', True, ('none',)),
    ('    return value', True, ('append_final_newline',)),
    ('    return value', False, ('none',)),
    ('```python\n    return value\n```\n', True, ('strip_single_python_fence',)),
    ('```\n    return value\n```', True, ('strip_single_python_fence_append_final_newline',)),
])
def test_v3_function_body_normalization_enum_is_conditioned_on_raw_body(
        raw, original_newline, expected):
    assert function_body_normalization_choices(
        raw, original_ends_with_newline=original_newline,
    ) == expected


def test_v3_function_body_normalization_does_not_promote_near_fences():
    raw = ' ```python\n    return value\n```\n'
    assert function_body_normalization_choices(
        raw, original_ends_with_newline=True,
    ) == ('none',)


def test_v4_forced_prefix_is_exact_and_ends_after_its_last_token():
    prefix = prefix_token_sequences(_ToyTokenizer(), '    ')
    assert prefix == (ord(' '),) * 4
    assert allowed_prefix_tokens([], prefix) == {ord(' ')}
    assert allowed_prefix_tokens([ord(' ')], prefix) == {ord(' ')}
    assert allowed_prefix_tokens(list(prefix), prefix) is None
    with pytest.raises(ValueError, match='diverged'):
        allowed_prefix_tokens([ord('x')], prefix)


@pytest.mark.parametrize(('text', 'expected'), [
    ('    return max(0, value - 1)', 'continue'),
    ('    return max(0, value - 1)\n', 'force_eos'),
    ('    return (\n', 'force_eos'),
    ('    return 1\n    return 2\n', 'invalid'),
])
def test_single_line_termination_state_forces_eos_only_after_one_complete_line(text, expected):
    assert single_line_termination_state(text) == expected


@pytest.mark.parametrize(('piece', 'expected'), [
    ('return value', True),
    (' return value\n', True),
    ('return value\nnext', False),
    ('return\n\n', False),
])
def test_single_line_token_mask_rejects_newline_tokens_with_trailing_text(piece, expected):
    assert single_line_token_piece_is_safe(piece) is expected


def test_forced_prefix_round_trip_is_required():
    class BrokenTokenizer(_ToyTokenizer):
        def decode(self, tokens, skip_special_tokens=False):
            return ''

    with pytest.raises(ValueError, match='does not round-trip'):
        prefix_token_sequences(BrokenTokenizer(), '    ')


def test_v6_function_controller_records_its_own_generation_protocol_source():
    sources = controller_source_hashes('lease_bound_single_line_function_edit_v1', 'schema_slots_v6')
    assert sources['src/vare/agent_function_edit_v1.py']
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


def test_transactional_edit_schema_constrains_normalization_to_declared_enum():
    tools = tools_for(
        'structured_replace', public_tests=True,
        tool_interface='lease_bound_transactional_navigation_v1',
    )
    tool = next(item for item in tools
                if item['function']['name'] == 'replace_lines_transactional')
    choices = tool['function']['parameters']['properties']['normalization']['enum']
    assert choices == [
        'none', 'strip_single_python_fence', 'append_final_newline',
        'strip_single_python_fence_append_final_newline',
    ]
    valid = {
        'lease': 'lease-v1', 'text': '    return 2\n', 'normalization': 'none',
    }
    encoded, payload = serialize_slot_action('replace_lines_transactional', valid, tools)
    assert parse_tool_output_v1(encoded, tools)[0] == 'complete_tool_call'
    assert payload['arguments'] == valid

    invalid = dict(valid, normalization='strip_everything_until_test_passes')
    with pytest.raises(SchemaSlotError, match='declared enum'):
        serialize_slot_action('replace_lines_transactional', invalid, tools)
    invalid_raw = (
        '<tool_call>{"name":"replace_lines_transactional","arguments":'
        '{"lease":"lease-v1","text":"    return 2\\n",'
        '"normalization":"strip_everything_until_test_passes"}}</tool_call>'
    )
    assert parse_tool_output_v1(invalid_raw, tools)[0] == 'invalid_json_or_schema'


def test_finite_string_enum_token_sequences_are_exact_and_round_trip():
    sequences = string_choice_token_sequences(_ToyTokenizer(), ['none', 'append_final_newline'])
    assert tuple(ord(char) for char in 'none') == sequences['none']
    assert tuple(ord(char) for char in 'append_final_newline') == sequences['append_final_newline']
    with pytest.raises(ValueError, match='nonempty and unique'):
        string_choice_token_sequences(_ToyTokenizer(), ['none', 'none'])
