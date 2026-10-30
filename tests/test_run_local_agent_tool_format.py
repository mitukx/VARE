import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from record_coding_agent_baseline import parse_diff
from verify_agent_workspace import verify_workspace
from scripts.run_local_agent import (
    MAX_TOOL_RESULT_BYTES,
    MAX_TOOL_RESULT_TOKENS,
    append_tool_interaction,
    attempt_configured_edit,
    bounded_list_page,
    bounded_search_results,
    bounded_source_page,
    bounded_tool_result,
    normalize_tool_call_format,
    render_chat_prompt,
    refresh_public_tests_after_edit,
    resolve_allowed_path,
    tools_for,
)
from scripts.run_local_agent import candidate_source_diff, snapshot_allowed_sources
from scripts.run_separated_coding_eval import (
    _attempted_task_run_count,
    _planned_task_run_count,
    verified_task_success,
)


def _git_workspace(tmp_path, relative_path, original):
    source = tmp_path / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(original)
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'config', 'user.name', 'VARE test'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'config', 'user.email', 'vare-test@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', '--', relative_path], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'commit', '--quiet', '-m', 'base'], check=True)
    revision = subprocess.run(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return source, revision


def _verify_candidate_diff(tmp_path, relative_path, original, replacement):
    source, base_revision = _git_workspace(tmp_path, relative_path, original)
    before = snapshot_allowed_sources(tmp_path, [relative_path])
    source.write_bytes(replacement)
    diff = candidate_source_diff(tmp_path, [relative_path], before, base_revision)
    source_before = {relative_path: hashlib.sha256(original).hexdigest()}
    source_after = {relative_path: hashlib.sha256(replacement).hexdigest()}
    agent = {'candidate_diff': diff, 'source_hashes_before': source_before,
             'source_hashes_after': source_after}
    grade = {'candidate_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
             'source_file_sha256': source_after}
    task = {'source': {'base_revision': base_revision, 'files': [relative_path]}}
    verify_workspace(tmp_path, task, agent, grade)
    assert parse_diff(diff, {relative_path}) == [relative_path]
    return diff


def _write_executable(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_candidate_diff_is_canonical_and_accepted_by_evidence_consumers(tmp_path):
    diff = _verify_candidate_diff(
        tmp_path, 'module.py', b'def f():\n    return 1\n', b'def f():\n    return 2\n',
    )

    assert diff.startswith('diff --git a/module.py b/module.py\n')


@pytest.mark.parametrize(
    ('original', 'replacement'),
    [(b'value=1', b'value=2'), (b'value=1\n', b'value=2\n'),
     (b'value=1\n', b'value=2'), (b'value=1', b'value=2\n')],
)
def test_candidate_diff_round_trips_terminal_newline_variants(tmp_path, original, replacement):
    diff = _verify_candidate_diff(tmp_path, 'module.py', original, replacement)
    if not original.endswith(b'\n') or not replacement.endswith(b'\n'):
        assert '\\ No newline at end of file' in diff


def test_candidate_diff_is_empty_when_workspace_is_unchanged(tmp_path):
    source, revision = _git_workspace(tmp_path, 'module.py', b'value=1\n')
    before = snapshot_allowed_sources(tmp_path, ['module.py'])

    assert candidate_source_diff(tmp_path, ['module.py'], before, revision) == ''


def test_candidate_diff_rejects_changed_workspace_head(tmp_path):
    source, revision = _git_workspace(tmp_path, 'module.py', b'value=1\n')
    before = snapshot_allowed_sources(tmp_path, ['module.py'])
    source.write_bytes(b'value=2\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'module.py'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'commit', '--quiet', '-m', 'tamper'], check=True)

    with pytest.raises(ValueError, match='HEAD changed'):
        candidate_source_diff(tmp_path, ['module.py'], before, revision)


def test_candidate_diff_ignores_ambient_git_overrides_external_diff_and_textconv(
    tmp_path, monkeypatch,
):
    source, base_revision = _git_workspace(tmp_path / 'candidate', 'module.py', b'value=1\n')
    before = snapshot_allowed_sources(source.parent, ['module.py'])
    source.write_bytes(b'value=2\n')

    other_root = tmp_path / 'other'
    _git_workspace(other_root, 'module.py', b'other=1\n')
    marker = tmp_path / 'git-hook-was-executed'
    external_diff = _write_executable(
        tmp_path / 'external-diff', f'echo invoked > "{marker}"\nprintf forged\n',
    )
    textconv = _write_executable(
        tmp_path / 'textconv', f'echo invoked > "{marker}"\nprintf forged\n',
    )
    (source.parent / '.git' / 'info' / 'attributes').write_text(
        'module.py diff=hostile\n', encoding='utf-8',
    )
    subprocess.run(
        ['git', '-C', str(source.parent), 'config', 'diff.hostile.textconv', str(textconv)],
        check=True,
    )
    monkeypatch.setenv('GIT_DIR', str(other_root / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(other_root))
    monkeypatch.setenv('GIT_INDEX_FILE', str(tmp_path / 'hostile-index'))
    monkeypatch.setenv('GIT_EXTERNAL_DIFF', str(external_diff))
    monkeypatch.setenv('GIT_REPLACE_REF_BASE', 'refs/hostile-replace/')
    monkeypatch.setenv('GIT_CONFIG_PARAMETERS', "'diff.external=/bin/false'")

    diff = candidate_source_diff(source.parent, ['module.py'], before, base_revision)

    assert diff.startswith('diff --git a/module.py b/module.py\n')
    assert '-value=1' in diff and '+value=2' in diff
    assert not marker.exists()
    after = {'module.py': hashlib.sha256(b'value=2\n').hexdigest()}
    agent = {
        'candidate_diff': diff,
        'source_hashes_before': {'module.py': hashlib.sha256(b'value=1\n').hexdigest()},
        'source_hashes_after': after,
    }
    grade = {
        'candidate_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'source_file_sha256': after,
    }
    task = {'source': {'base_revision': base_revision, 'files': ['module.py']}}
    assert verify_workspace(source.parent, task, agent, grade)['workspace_revision'] == base_revision
    assert not marker.exists()


def test_candidate_diff_rejects_symlink_source_file_and_parent(tmp_path):
    source, base_revision = _git_workspace(tmp_path / 'file-link', 'module.py', b'value=1\n')
    outside = tmp_path / 'outside.py'
    outside.write_bytes(b'value=1\n')
    source.unlink()
    source.symlink_to(outside)

    with pytest.raises(ValueError, match='linked|symlink|source path'):
        snapshot_allowed_sources(source.parent, ['module.py'])

    nested, nested_revision = _git_workspace(
        tmp_path / 'parent-link', 'pkg/module.py', b'value=1\n',
    )
    nested_root = nested.parent.parent
    nested_before = snapshot_allowed_sources(nested_root, ['pkg/module.py'])
    outside_dir = tmp_path / 'outside-dir'
    outside_dir.mkdir()
    (outside_dir / 'module.py').write_bytes(b'value=2\n')
    (nested_root / 'pkg').rename(nested_root / 'pkg-original')
    (nested_root / 'pkg').symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match='linked|symlink|source path'):
        candidate_source_diff(
            nested_root, ['pkg/module.py'], nested_before, nested_revision,
        )


def test_candidate_diff_rejects_a_symlinked_git_directory(tmp_path):
    source, base_revision = _git_workspace(tmp_path / 'candidate', 'module.py', b'value=1\n')
    root = source.parent
    before = snapshot_allowed_sources(root, ['module.py'])
    git_dir = root / '.git'
    moved_git_dir = tmp_path / 'candidate-git-metadata'
    git_dir.rename(moved_git_dir)
    git_dir.symlink_to(moved_git_dir, target_is_directory=True)

    with pytest.raises(ValueError, match='Git metadata|workspace'):
        candidate_source_diff(root, ['module.py'], before, base_revision)


def test_candidate_diff_ignores_replace_refs_and_rebuilds_a_pinned_index(tmp_path):
    source, base_revision = _git_workspace(tmp_path, 'module.py', b'value=1\n')
    before = snapshot_allowed_sources(tmp_path, ['module.py'])
    replacement_blob = subprocess.run(
        ['git', '-C', str(tmp_path), 'hash-object', '-w', '--stdin'],
        input=b'value=2\n', check=True, capture_output=True,
    ).stdout.decode().strip()
    subprocess.run(
        ['git', '-C', str(tmp_path), 'update-index', '--cacheinfo',
         f'100644,{replacement_blob},module.py'], check=True,
    )
    replacement_tree = subprocess.run(
        ['git', '-C', str(tmp_path), 'write-tree'], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    replacement_commit = subprocess.run(
        ['git', '-C', str(tmp_path), 'commit-tree', replacement_tree,
         '-p', base_revision, '-m', 'replacement object'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ['git', '-C', str(tmp_path), 'read-tree', base_revision], check=True,
    )
    subprocess.run(
        ['git', '-C', str(tmp_path), 'replace', base_revision, replacement_commit],
        check=True,
    )
    source.write_bytes(b'value=2\n')

    diff = candidate_source_diff(tmp_path, ['module.py'], before, base_revision)

    assert '-value=1' in diff and '+value=2' in diff
    after = {'module.py': hashlib.sha256(b'value=2\n').hexdigest()}
    agent = {
        'candidate_diff': diff,
        'source_hashes_before': {'module.py': hashlib.sha256(b'value=1\n').hexdigest()},
        'source_hashes_after': after,
    }
    grade = {
        'candidate_diff_sha256': hashlib.sha256(diff.encode()).hexdigest(),
        'source_file_sha256': after,
    }
    task = {'source': {'base_revision': base_revision, 'files': ['module.py']}}
    assert verify_workspace(tmp_path, task, agent, grade)['workspace_revision'] == base_revision


def test_candidate_diff_rejects_source_drift_before_model_invocation(tmp_path):
    source, base_revision = _git_workspace(tmp_path, 'module.py', b'value=1\n')
    source.write_bytes(b'drifted=1\n')
    drifted_before = snapshot_allowed_sources(tmp_path, ['module.py'])

    with pytest.raises(ValueError, match='snapshot differs from pinned Git blob'):
        candidate_source_diff(tmp_path, ['module.py'], drifted_before, base_revision)


def test_candidate_diff_rejects_staged_or_untracked_workspace_changes(tmp_path):
    source, base_revision = _git_workspace(tmp_path, 'module.py', b'value=1\n')
    before = snapshot_allowed_sources(tmp_path, ['module.py'])
    source.write_bytes(b'value=2\n')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'module.py'], check=True)

    with pytest.raises(ValueError, match='staged or non-allowlisted'):
        candidate_source_diff(tmp_path, ['module.py'], before, base_revision)


def test_normalizes_a_fenced_qwen_json_tool_call_without_changing_raw_text():
    raw = '```json\n{"name":"list_files","arguments":{}}\n```'

    normalized, recovered = normalize_tool_call_format(raw, {"list_files", "read_file"})

    assert recovered is True
    assert normalized == '<tool_call>{"name":"list_files","arguments":{}}</tool_call>'
    assert raw.startswith("```json")


def test_normalizes_a_bare_qwen_json_tool_call():
    raw = '{"name":"read_file","arguments":{"path":"grad_accum.py"}}'
    assert normalize_tool_call_format(raw, {"read_file"}) == (
        '<tool_call>{"name":"read_file","arguments":{"path":"grad_accum.py"}}</tool_call>',
        True,
    )


def test_normalizes_a_chatml_wrapped_json_tool_call():
    raw = '<|im_start|>\n{"name":"write_file","arguments":{"path":"x.py","content":"pass\\n"}}\n<|im_end|>'
    assert normalize_tool_call_format(raw, {"write_file"}) == (
        '<tool_call>{"name":"write_file","arguments":{"path":"x.py","content":"pass\\n"}}</tool_call>',
        True,
    )


def test_read_file_schema_exposes_bounded_inclusive_line_ranges():
    read_file = next(
        item['function'] for item in tools_for('unified_diff')
        if item['function']['name'] == 'read_file'
    )
    parameters = read_file['parameters']

    assert parameters['required'] == ['path']
    assert parameters['properties']['start_line'] == {'type': 'integer', 'minimum': 1}
    assert parameters['properties']['end_line'] == {'type': 'integer', 'minimum': 1}
    assert parameters['properties']['char_offset'] == {'type': 'integer', 'minimum': 0}
    assert '12,000-byte' in read_file['description']


def test_bounded_source_page_defaults_to_180_lines_and_reports_more():
    source = ''.join(f'line-{number}\n' for number in range(1, 451))

    page = bounded_source_page(source)

    assert page['content'] == ''.join(f'line-{number}\n' for number in range(1, 181))
    assert page['start_line'] == 1
    assert page['end_line'] == 180
    assert page['total_lines'] == 450
    assert page['has_more'] is True
    assert page['next_cursor'] == {'start_line': 181, 'char_offset': 0}


def test_bounded_source_page_reads_requested_inclusive_range():
    source = ''.join(f'line-{number}\n' for number in range(1, 451))

    page = bounded_source_page(source, start_line=201, end_line=220)

    assert page['content'] == ''.join(f'line-{number}\n' for number in range(201, 221))
    assert page['start_line'] == 201
    assert page['end_line'] == 220
    assert page['total_lines'] == 450
    assert page['has_more'] is True
    assert page['next_cursor'] == {'start_line': 221, 'char_offset': 0}


def test_bounded_source_page_caps_requests_at_200_lines():
    source = ''.join(f'line-{number}\n' for number in range(1, 451))

    page = bounded_source_page(source, end_line=450)

    assert page['content'] == ''.join(f'line-{number}\n' for number in range(1, 201))
    assert page['start_line'] == 1
    assert page['end_line'] == 200
    assert page['total_lines'] == 450
    assert page['has_more'] is True
    assert page['next_cursor'] == {'start_line': 201, 'char_offset': 0}


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def test_bounded_source_page_enforces_token_budget_and_continues_long_line():
    tokenizer = _CharacterTokenizer()
    source = 'x' * 1_200 + '\nnext line\n'
    identity = {'resolved_path': 'pkg/source.py'}
    page = bounded_source_page(
        source, tokenizer=tokenizer, source_identity=identity,
        max_output_bytes=10_000, max_output_tokens=240,
    )
    encoded = json.dumps(page, ensure_ascii=False, separators=(',', ':'))
    assert len(encoded.encode()) <= 10_000
    assert len(tokenizer.encode(encoded)) <= 240
    assert page['resolved_path'] == 'pkg/source.py'
    assert page['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    assert page['start_line'] == page['end_line'] == 1
    assert page['has_more'] is True
    assert page['next_cursor']['start_line'] == 1
    assert page['next_cursor']['char_offset'] == len(page['content'])

    cursor = page['next_cursor']
    continuation = bounded_source_page(
        source, start_line=cursor['start_line'], char_offset=cursor['char_offset'],
        tokenizer=tokenizer, source_identity=identity,
        max_output_bytes=10_000, max_output_tokens=240,
    )
    assert continuation['content']
    assert continuation['source_sha256'] == page['source_sha256']


def test_search_and_file_listing_are_tokenizer_bounded_and_paginated():
    tokenizer = _CharacterTokenizer()
    matches = [{'path': f'src/file_{i}.py', 'line': i + 1, 'text': 'match ' * 100}
               for i in range(40)]
    result = bounded_search_results(
        matches, tokenizer=tokenizer, max_output_bytes=10_000,
        max_output_tokens=500,
    )
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    assert len(encoded.encode()) <= 10_000
    assert len(tokenizer.encode(encoded)) <= 500
    assert result['matches'][0]['path'] == 'src/file_0.py'
    assert result['has_more'] is True
    assert result['next_offset'] == len(result['matches'])
    assert bounded_search_results(matches, offset=result['next_offset'])['matches'][0]['line'] == result['next_offset'] + 1

    files = bounded_list_page(
        [f'src/module_{i:03}.py' for i in range(200)], tokenizer=tokenizer,
        max_output_tokens=180,
    )
    assert len(tokenizer.encode(json.dumps(files, separators=(',', ':')))) <= 180
    assert files['has_more'] is True
    assert files['next_offset'] == len(files['files'])


def test_tool_result_final_cap_and_native_template_keep_multiturn_continuity():
    tokenizer = _CharacterTokenizer()
    oversized = bounded_tool_result({'stdout': 'z' * 20_000}, tokenizer)
    encoded = json.dumps(oversized, ensure_ascii=False, separators=(',', ':'))
    assert len(encoded.encode()) <= MAX_TOOL_RESULT_BYTES
    assert len(tokenizer.encode(encoded)) <= MAX_TOOL_RESULT_TOKENS
    assert oversized['response_truncated'] is True

    class RecordingTokenizer(_CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return 'native-template-prompt'

    recording = RecordingTokenizer()
    messages = [{'role': 'user', 'content': 'inspect and edit'}]
    append_tool_interaction(messages, [
        ('read_file', {'path': 'pkg/source.py'}, {'content': 'value = 1', 'start_line': 1}),
        ('apply_patch', {'patch': '...'}, {'applied': True, 'paths': ['pkg/source.py']}),
    ])
    prompt = render_chat_prompt(recording, messages, tools_for('unified_diff'))
    assert prompt == 'native-template-prompt'
    assert [message['role'] for message in recording.messages] == [
        'user', 'assistant', 'tool', 'tool',
    ]
    assert recording.messages[2]['name'] == 'read_file'
    assert recording.messages[3]['name'] == 'apply_patch'
    assert recording.messages[1]['tool_calls'][0]['id'] == recording.messages[2]['tool_call_id']
    assert recording.messages[1]['tool_calls'][1]['id'] == recording.messages[3]['tool_call_id']
    assert recording.kwargs['tools'] == tools_for('unified_diff')


def test_source_lookup_and_patch_application_fixture(tmp_path):
    source, _revision = _git_workspace(tmp_path, 'pkg/source.py', b'value = 1\n')
    assert resolve_allowed_path('/candidate/pkg/source.py', {'pkg/source.py'}) == 'pkg/source.py'
    patch = (
        '--- a/pkg/source.py\n+++ b/pkg/source.py\n'
        '@@ -1 +1 @@\n-value = 1\n+value = 2\n'
    )
    applied = attempt_configured_edit(tmp_path, patch, 'unified_diff', {'pkg/source.py'})
    assert applied == {'applied': True, 'paths': ['pkg/source.py'], 'format_parse_failure': None}
    assert source.read_text() == 'value = 2\n'


@pytest.mark.parametrize(
    ('kwargs', 'message'),
    [
        ({'start_line': 0}, 'start_line must be a positive integer'),
        ({'start_line': True}, 'start_line must be a positive integer'),
        ({'end_line': 0}, 'end_line must be a positive integer'),
        ({'start_line': 4, 'end_line': 3}, 'end_line must be greater than or equal to start_line'),
        ({'start_line': 4}, 'start_line exceeds the source length'),
    ],
)
def test_bounded_source_page_rejects_invalid_ranges(kwargs, message):
    with pytest.raises(ValueError, match=message):
        bounded_source_page('one\ntwo\nthree\n', **kwargs)


def test_bounded_source_page_handles_empty_files():
    assert bounded_source_page('') == {
        'source_sha256': hashlib.sha256(b'').hexdigest(),
        'content': '', 'start_line': 1, 'end_line': 0, 'total_lines': 0,
        'has_more': False, 'next_cursor': None,
    }


def test_normalizes_json_before_a_trailing_chatml_generation_marker():
    raw = '<|im_start|>\n{"name":"list_files","arguments":{}}\n<|im_start|>\n'
    assert normalize_tool_call_format(raw, {"list_files"}) == (
        '<tool_call>{"name":"list_files","arguments":{}}</tool_call>', True,
    )


def test_does_not_promote_unknown_or_malformed_fenced_json_to_tool_calls():
    for raw in (
        '```json\n{"name":"run_shell","arguments":{}}\n```',
        '```json\n{"name":"read_file","arguments":[]}\n```',
        '```python\nprint("not a tool")\n```',
        '{"name":"run_shell","arguments":{}}',
    ):
        assert normalize_tool_call_format(raw, {"read_file"}) == (raw, False)


def test_preserves_native_tool_call_format():
    raw = '<tool_call>{"name":"finish","arguments":{"summary":"done"}}</tool_call>'
    assert normalize_tool_call_format(raw, {"finish"}) == (raw, False)


def test_verified_success_requires_public_test_and_independent_hidden_grade():
    agent = {
        "source_hashes_before": {"x.py": "before"},
        "source_hashes_after": {"x.py": "after"},
        "public_test_result": {"passed": True},
    }
    assert verified_task_success(agent, {"success": True}) is True
    assert verified_task_success({**agent, "public_test_result": None}, {"success": True}) is False
    assert verified_task_success(agent, {"success": False}) is False


def test_applied_recovered_edit_requires_a_fresh_public_test(monkeypatch, tmp_path):
    calls = []

    def fail_fresh_test(workspace, image, test_hash):
        calls.append((workspace, image, test_hash))
        return {"passed": False, "returncode": 1}

    monkeypatch.setattr("scripts.run_local_agent.run_public_tests", fail_fresh_test)
    stale_result = {"passed": True, "returncode": 0}
    refreshed = refresh_public_tests_after_edit(
        {"applied": True}, tmp_path, "pinned-image", "test-hash",
    )
    assert refreshed == {"passed": False, "returncode": 1}
    assert refreshed is not stale_result
    assert calls == [(tmp_path, "pinned-image", "test-hash")]


def test_attempted_task_run_count_includes_every_arm_and_secondary_model():
    arm_reports = [
        {"name": "greedy", "rows": [{}, {}]},
        {"name": "sampled", "rows": [{}, {}]},
    ]
    secondary = {"rows": [{}], "coder_1_5b_rows": [{}, {}]}
    assert _attempted_task_run_count(arm_reports, secondary) == 7


def test_planned_task_run_count_matches_admitted_secondary_models():
    protocol = {"models": {
        "secondary_if_feasible": {"seeds": [201]},
        "not_run_unless": {"seeds": [301, 302]},
    }}
    arms = [{"seeds": [101, 102]}]
    assert _planned_task_run_count(
        protocol, arms, 3, {"coder_0_5b": "eligible", "coder_1_5b": "ran"},
    ) == 15
    assert _planned_task_run_count(
        protocol, arms, 3, {"coder_0_5b": "skipped", "coder_1_5b": "skipped"},
    ) == 6
