from __future__ import annotations

import json

from scripts.run_local_agent import (
    candidate_visible_tool_result,
    bounded_turn_payload,
    classify_public_test_error,
    classify_public_test_exit,
    no_tool_retry_reminder,
    public_test_infrastructure_error,
    record_automatic_public_test,
    record_tool_call,
    should_stop_after_tool,
    tools_for,
)


def _tool(name: str, *, opaque_handles: bool = True) -> dict[str, object]:
    return next(
        tool['function']
        for tool in tools_for('structured', public_tests=True, opaque_handles=opaque_handles)
        if tool['function']['name'] == name
    )


def test_opaque_structured_edit_schema_does_not_request_long_hashes() -> None:
    read = _tool('read_file')
    replace = _tool('replace_text')
    insert = _tool('insert_text')
    assert read['parameters']['required'] == ['path']
    assert 'workspace_revision_handle' in replace['parameters']['properties']
    assert 'source_preimage_handle' in replace['parameters']['properties']
    assert 'workspace_revision_handle' not in replace['parameters']['required']
    assert 'source_preimage_handle' not in replace['parameters']['required']
    assert 'expected_hash' not in replace['parameters']['properties']
    assert 'expected_revision' not in replace['parameters']['properties']
    assert 'workspace_revision_handle' not in insert['parameters']['required']
    assert 'source_preimage_handle' not in insert['parameters']['required']


def test_legacy_structured_schema_remains_available_without_opaque_mode() -> None:
    replace = _tool('replace_text', opaque_handles=False)
    assert 'expected_hash' in replace['parameters']['properties']
    assert 'expected_revision' in replace['parameters']['properties']
    assert 'workspace_revision_handle' not in replace['parameters']['properties']


def test_model_feedback_omits_host_hashes_and_bounds_public_assertion() -> None:
    source_hash = 'a' * 64
    failure_signature = 'b' * 64
    result = {
        'applied': True,
        'path': 'src/module.py',
        'source_sha256_before': source_hash,
        'workspace_revision_before': source_hash,
        'automatic_public_test': {
            'passed': False,
            'failure_signature': failure_signature,
            'feedback': 'assert expected behavior',
            'next_action': 'inspect the failing assertion and make a different targeted edit',
        },
    }
    visible = candidate_visible_tool_result('replace_text', result, opaque_handles=True)
    assert source_hash not in str(visible)
    assert failure_signature not in str(visible)
    assert visible['automatic_public_test']['feedback'] == 'assert expected behavior'
    assert visible['automatic_public_test']['next_action'].startswith('inspect')


def test_model_does_not_receive_internal_diff_or_test_hashes() -> None:
    result = {'diff': 'small patch', 'diff_sha256': 'a' * 64, 'changed': True}
    visible = candidate_visible_tool_result('inspect_diff', result, opaque_handles=True)
    assert visible == {'diff': 'small patch', 'changed': True}
    public = {'passed': False, 'failure_signature': 'b' * 64, 'feedback': 'failure', 'next_action': 'repair'}
    visible_test = candidate_visible_tool_result('run_public_tests', public, opaque_handles=True)
    assert 'failure_signature' not in visible_test
    assert visible_test['feedback'] == 'failure'


def test_public_test_exit_codes_separate_launch_failures_from_candidate_failures() -> None:
    assert classify_public_test_exit(0) == 'passed'
    assert classify_public_test_exit(1) == 'test_failure'
    assert classify_public_test_exit(125) == 'container_launch_failure'
    assert classify_public_test_exit(126) == 'container_launch_failure'
    assert classify_public_test_exit(127) == 'container_launch_failure'
    assert classify_public_test_exit(137) == 'container_resource_failure'
    assert classify_public_test_error('IsolationError: container exceeded wall-time limit') == 'candidate_timeout'
    assert public_test_infrastructure_error({'classification': 'container_launch_failure'})
    assert not public_test_infrastructure_error({'classification': 'test_failure'})


def test_public_test_feedback_reports_infrastructure_status_without_host_path() -> None:
    visible = candidate_visible_tool_result(
        'run_public_tests',
        {
            'passed': False,
            'classification': 'container_launch_failure',
            'feedback': 'Public test execution was unavailable; no test assertion ran.',
            'next_action': 'stop this rollout; the evaluator must be repaired before grading',
            'stderr': 'bind source path does not exist: /private/var/folders/secret',
        },
        opaque_handles=True,
    )
    assert visible['classification'] == 'container_launch_failure'
    assert 'no test assertion ran' in visible['feedback']
    assert 'secret' not in str(visible)


def test_rejected_finalization_keeps_repair_loop_open() -> None:
    assert should_stop_after_tool('finish', {'finished': False, 'error': 'tests have not passed'}) is False
    assert should_stop_after_tool('finish', {'finished': True}) is True
    assert should_stop_after_tool('inspect_diff', {'finished': True}) is False


def test_candidate_diff_feedback_is_bounded_but_transcript_keeps_full_diff() -> None:
    diff = 'a' * 10_000
    raw = {'diff': diff, 'changed': True, 'diff_sha256': 'full-digest'}
    visible = candidate_visible_tool_result('inspect_diff', raw, opaque_handles=True)
    assert len(visible['diff']) < 6_000
    assert 'diff truncated for model context' in visible['diff']
    assert 'chars=10000' in visible['diff']
    assert raw['diff'] == diff


def test_edit_tool_diff_feedback_is_bounded() -> None:
    raw = {'applied': True, 'path': 'src/a.py', 'diff': 'b' * 10_000}
    visible = candidate_visible_tool_result('replace_text', raw, opaque_handles=True)
    assert len(visible['diff']) < 6_000
    assert 'diff truncated for model context' in visible['diff']


def test_opaque_read_feedback_uses_per_turn_cumulative_budget() -> None:
    used = 0
    outputs = []
    for content in ('a' * 5_000, 'b' * 5_000, 'c' * 5_000):
        visible = candidate_visible_tool_result(
            'read_file',
            {
                'path': 'src/a.py',
                'content': content,
                'workspace_revision_handle': 'rev-0001',
                'source_preimage_handle': 'src-0001',
                'private_sha256': 'must not reach model',
            },
            opaque_handles=True,
            max_read_chars=max(0, 6_000 - used),
        )
        outputs.append(visible)
        used += len(visible['content'])
        assert 'private_sha256' not in visible
    assert used <= 6_000
    assert len(outputs[0]['content']) == 5_000
    assert outputs[1]['truncated'] is True
    assert outputs[2]['content'] == ''


def test_truncated_opaque_read_reports_only_complete_lines_and_next_line() -> None:
    content = ''.join(f'line-{index:03d}\n' for index in range(1, 121))
    visible = candidate_visible_tool_result(
        'read_file',
        {
            'path': 'src/a.py', 'start_line': 40, 'end_line': 159,
            'content': content, 'workspace_revision_handle': 'rev-1',
            'source_preimage_handle': 'src-1',
        },
        opaque_handles=True,
        max_read_chars=240,
    )
    returned_lines = visible['content'].splitlines()
    assert visible['truncated'] is True
    assert returned_lines
    assert visible['end_line'] == 39 + len(returned_lines)
    assert visible['next_line'] == visible['end_line'] + 1
    assert len(json.dumps(visible, ensure_ascii=False)) <= 240


def test_opaque_search_feedback_uses_same_cumulative_budget() -> None:
    results = [
        {'path': 'src/a.py', 'line': index, 'text': 'x' * 300}
        for index in range(30)
    ]
    first = candidate_visible_tool_result(
        'search_text', results, opaque_handles=True, max_read_chars=2_000,
    )
    used = len(json.dumps(first, ensure_ascii=False))
    second = candidate_visible_tool_result(
        'search_text', results, opaque_handles=True, max_read_chars=2_000 - used,
    )
    assert used <= 2_000
    assert len(json.dumps(second, ensure_ascii=False)) <= 2_000 - used
    assert len(first) < len(results)


def test_manual_test_transcript_keeps_raw_result_when_feedback_is_summarized() -> None:
    raw = {
        'passed': False,
        'returncode': 1,
        'stdout': 'collected 4 tests',
        'stderr': 'AssertionError: expected value',
        'elapsed_seconds': 1.25,
        'failure_signature': 'private-to-controller',
    }
    entry: dict[str, object] = {'tool_calls': []}
    record_tool_call(entry, 'run_public_tests', {}, raw)
    visible = candidate_visible_tool_result(
        'run_public_tests',
        {'passed': False, 'feedback': 'assertion failed', 'next_action': 'repair'},
        opaque_handles=True,
    )
    assert visible == {'passed': False, 'feedback': 'assertion failed', 'next_action': 'repair'}
    assert entry['tool_calls'][0]['result'] == raw


def test_automatic_public_test_history_retains_every_raw_result() -> None:
    entry: dict[str, object] = {}
    first = {'passed': False, 'returncode': 1, 'stdout': 'first', 'stderr': 'assertion one'}
    second = {'passed': True, 'returncode': 0, 'stdout': 'second', 'stderr': ''}
    record_automatic_public_test(entry, first)
    record_automatic_public_test(entry, second)
    assert entry['automatic_public_tests'] == [first, second]
    assert entry['automatic_public_test'] == second


def test_shared_turn_payload_budget_caps_edit_and_diff_feedback() -> None:
    result = {'diff': 'x' * 7_000, 'applied': True}
    visible = candidate_visible_tool_result('inspect_diff', result, opaque_handles=True)
    payload = bounded_turn_payload(visible, 2_000)
    assert len(payload) <= 2_000
    assert len(bounded_turn_payload({'passed': False}, 2_000 - len(payload))) <= 2_000 - len(payload)
    assert bounded_turn_payload({'passed': True}, 0) == ''


def test_large_diff_keeps_head_and_tail_excerpt_inside_serialized_budget() -> None:
    diff = ''.join(f'- removed line {index}\n' for index in range(8_000))
    encoded = bounded_turn_payload({'diff': diff, 'changed': True}, 5_900)
    visible = json.loads(encoded)
    assert len(encoded) <= 5_900
    assert visible['diff_truncated'] is True
    assert visible['diff'].startswith('- removed line 0')
    assert visible['diff'].rstrip().endswith(f'- removed line {7_999}')
    assert visible['diff_sha256']


def test_manual_public_test_feedback_omits_internal_failure_signature() -> None:
    raw = {
        'passed': False,
        'feedback': 'assertion failed',
        'next_action': 'inspect the relevant branch',
        'failure_signature': 'private-to-controller',
    }
    visible = candidate_visible_tool_result('run_public_tests', raw, opaque_handles=True)
    assert visible == {
        'passed': False,
        'feedback': 'assertion failed',
        'next_action': 'inspect the relevant branch',
    }


def test_truncated_structured_retry_uses_active_opaque_handle_names() -> None:
    reminder = no_tool_retry_reminder(
        '<tool_call>{"name":"replace_text"',
        edit_format='structured',
        allowed_paths=['src/a.py'],
        token_limit=512,
        opaque_handles=True,
    )
    assert 'workspace_revision_handle' in reminder
    assert 'source_preimage_handle' in reminder
    assert 'expected_hash' not in reminder


def test_edit_tool_descriptions_match_active_handle_schema() -> None:
    opaque = {tool['function']['name']: tool['function']['description']
              for tool in tools_for('structured', public_tests=True, opaque_handles=True)}
    legacy = {tool['function']['name']: tool['function']['description']
              for tool in tools_for('structured', public_tests=False, opaque_handles=False)}
    for name in ('replace_text', 'insert_text'):
        assert 'workspace_revision_handle' in opaque[name]
        assert 'expected_hash' not in opaque[name]
        assert 'expected_hash' in legacy[name]
        assert 'workspace_revision_handle' not in legacy[name]
