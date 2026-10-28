import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from record_coding_agent_baseline import parse_diff
from verify_agent_workspace import verify_workspace
from scripts.run_local_agent import normalize_tool_call_format, refresh_public_tests_after_edit
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
