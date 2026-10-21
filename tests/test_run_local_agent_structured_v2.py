from __future__ import annotations

from scripts.run_local_agent import (
    candidate_visible_tool_result,
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
    assert 'expected_hash' not in replace['parameters']['properties']
    assert 'expected_revision' not in replace['parameters']['properties']
    assert 'workspace_revision_handle' in insert['parameters']['required']
    assert 'source_preimage_handle' in insert['parameters']['required']


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


def test_rejected_finalization_keeps_repair_loop_open() -> None:
    assert should_stop_after_tool('finish', {'finished': False, 'error': 'tests have not passed'}) is False
    assert should_stop_after_tool('finish', {'finished': True}) is True
    assert should_stop_after_tool('inspect_diff', {'finished': True}) is False
