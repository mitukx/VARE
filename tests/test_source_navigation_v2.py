from pathlib import Path

from scripts.run_local_agent import controller_source_hashes, tools_for
from vare.source_navigation_v1 import SourceNavigatorV1
from vare.source_navigation_v2 import SourceNavigatorV2


def test_qualified_class_method_search_resolves_across_source_lines(tmp_path: Path):
    source = tmp_path / 'module.py'
    source.write_text(
        'class CachedPositionEmbedding:\n'
        '    def __init__(self):\n'
        '        self.weights = []\n'
        '\n'
        '    def forward(self, seq_len, past_key_values_length=0):\n'
        '        needed = seq_len + past_key_values_length\n'
        '        return needed\n',
        encoding='utf-8',
    )
    base = SourceNavigatorV1(tmp_path, {'module.py'}).search('CachedPositionEmbedding.forward')
    enhanced = SourceNavigatorV2(tmp_path, {'module.py'}).search('CachedPositionEmbedding.forward')

    assert base['results'] == []
    assert enhanced['qualified_name_resolution'] == 'ast_class_method'
    assert enhanced['results'][0]['path'] == 'module.py'
    assert enhanced['results'][0]['line'] == 5
    assert enhanced['results'][0]['match_kind'] == 'python_class_method'


def test_v2_falls_back_to_bounded_v1_text_search(tmp_path: Path):
    (tmp_path / 'module.py').write_text('cache_position = 3\n', encoding='utf-8')
    result = SourceNavigatorV2(tmp_path, {'module.py'}).search('cache_position')
    assert result['results'][0]['line'] == 1
    assert 'qualified_name_resolution' not in result


def test_v2_rejects_repeated_query_on_unchanged_source_snapshot(tmp_path: Path):
    (tmp_path / 'module.py').write_text(
        'def update_cache(cache_position):\n'
        '    return cache_position\n',
        encoding='utf-8',
    )
    navigator = SourceNavigatorV2(tmp_path, {'module.py'})

    first = navigator.search('update_cache')
    repeated = navigator.search('  UPDATE_CACHE  ')

    assert first['status'] == 'ok'
    assert first['results']
    assert repeated['status'] == 'duplicate_query'
    assert repeated['index_sha256'] == first['index_sha256']
    assert repeated['index_generation'] == first['index_generation']
    assert 'shorter code identifier' in repeated['retry_hint']


def test_v2_allows_same_query_after_source_snapshot_changes(tmp_path: Path):
    source = tmp_path / 'module.py'
    source.write_text('def update_cache():\n    return 1\n', encoding='utf-8')
    navigator = SourceNavigatorV2(tmp_path, {'module.py'})

    first = navigator.search('update_cache')
    source.write_text('def update_cache():\n    return 2\n', encoding='utf-8')
    after_change = navigator.search('update_cache')

    assert first['status'] == 'ok'
    assert after_change['status'] == 'ok'
    assert after_change['index_generation'] > first['index_generation']
    assert after_change['index_sha256'] != first['index_sha256']
    assert after_change['index_rebuilt'] is True


def test_v2_interface_hashes_both_navigator_versions_and_keeps_tool_contract():
    sources = controller_source_hashes('lease_bound_navigation_v2', 'schema_slots_v1')
    assert sources['src/vare/source_navigation_v1.py']
    assert sources['src/vare/source_navigation_v2.py']
    transactional = controller_source_hashes(
        'lease_bound_transactional_navigation_v1', 'schema_slots_v6',
    )
    assert transactional['src/vare/source_navigation_v1.py']
    assert transactional['src/vare/source_navigation_v2.py']
    assert transactional['src/vare/agent_function_edit_v1.py']
    assert transactional['src/vare/agent_json_eos_v1.py']
    assert transactional['src/vare/agent_json_eos_v2.py']
    assert transactional['src/vare/agent_json_eos_v3.py']
    assert transactional['src/vare/schema_slot_protocol.py']
    v1 = tools_for('structured_replace', public_tests=True,
                   tool_interface='lease_bound_navigation_v1')
    v2 = tools_for('structured_replace', public_tests=True,
                   tool_interface='lease_bound_navigation_v2')
    assert v1 == v2

