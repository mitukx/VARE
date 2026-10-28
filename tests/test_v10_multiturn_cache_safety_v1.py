"""CPU safety checks for the v9/v10 multi-turn shared-prefix benchmark.

These use fake MLX modules and character tokenizers. They check controller
cache semantics and benchmark instrumentation, not real-model latency.
"""
from __future__ import annotations

import gc
import json
import sys
import weakref
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import run_local_agent
from scripts.analyze_v10_multiturn_v1 import interval
from scripts.run_v10_multiturn_worker_v1 import (
    TOOL_INTERFACE, SyntheticWorkspace, audit_shared_prefix, initial_messages, tools_for_turn,
)
from vare.synthetic_function_tests_v1 import run_function_tests

EVIDENCE = Path(__file__).resolve().parents[1] / 'results' / 'v10_multiturn_abba_20261028'


class CharTokenizer:
    bos_token = None
    eos_token_id = 99
    eos_token_ids = [99]

    def __init__(self, boundary_mismatch_turns=()):
        self.boundary_mismatch_turns = set(boundary_mismatch_turns)
        self.turn = 0
        self.shared_prefix = None

    @staticmethod
    def render(messages):
        parts = []
        for item in messages:
            body = item.get('content')
            if body is None:
                body = json.dumps(item.get('tool_calls'), sort_keys=True)
            parts.append(f"<{item['role']}>{body}</{item['role']}>")
        return ''.join(parts)

    def apply_chat_template(self, messages, **kwargs):
        rendered = self.render(messages)
        if kwargs.get('add_generation_prompt'):
            return rendered + '<assistant>'
        self.shared_prefix = rendered
        return rendered

    def encode(self, text, add_special_tokens=False):
        ids = [ord(char) for char in text]
        if (self.turn in self.boundary_mismatch_turns and self.shared_prefix
                and text != self.shared_prefix and text.startswith(self.shared_prefix)):
            ids[len(self.shared_prefix) - 1] = 0x10FFFF
        return ids

    @staticmethod
    def decode(tokens, skip_special_tokens=False):
        return ''.join(chr(token) for token in tokens if token < 0x110000)


@pytest.fixture
def fake_mlx(monkeypatch):
    state = {'outputs': [], 'calls': [], 'caches': [], 'prefix_builds': []}
    core = ModuleType('mlx.core')
    core.array = deepcopy
    core.copy = deepcopy
    core.eval = lambda *values: None
    core.stream = lambda _stream: nullcontext()
    core.reset_peak_memory = lambda: None
    core.get_active_memory = lambda: 10
    core.get_peak_memory = lambda: 20
    core.metal = SimpleNamespace(get_active_memory=lambda: 10, get_peak_memory=lambda: 20)
    mlx = ModuleType('mlx')
    mlx.__path__ = []
    mlx.core = core

    class Cache:
        def __init__(self):
            self.state = ([],)
            state['caches'].append(weakref.ref(self))

    cache_module = ModuleType('mlx_lm.models.cache')
    cache_module.make_prompt_cache = lambda model: [Cache()]
    generate = ModuleType('mlx_lm.generate')
    generate.generation_stream = object()

    def stream_generate(model, tokenizer, *, prompt, max_tokens, sampler, logits_processors, prompt_cache=None):
        text = state['outputs'].pop(0)
        state['calls'].append({'prompt': list(prompt),
                               'initial_cache': deepcopy(prompt_cache[0].state[0]) if prompt_cache else None})
        if prompt_cache is not None:
            prompt_cache[0].state = (prompt_cache[0].state[0] + list(prompt),)
        yield SimpleNamespace(text=text, prompt_tokens=len(prompt), prompt_tps=100.0,
                              generation_tokens=len(text), finish_reason='length')

    generate.stream_generate = stream_generate
    sample_utils = ModuleType('mlx_lm.sample_utils')
    sample_utils.make_sampler = lambda **kwargs: 'fixture-sampler'
    mlx_lm = ModuleType('mlx_lm')
    mlx_lm.__path__ = []
    models = ModuleType('mlx_lm.models')
    models.__path__ = []
    for name, module in (('mlx', mlx), ('mlx.core', core), ('mlx_lm', mlx_lm), ('mlx_lm.generate', generate),
                         ('mlx_lm.models', models), ('mlx_lm.models.cache', cache_module),
                         ('mlx_lm.sample_utils', sample_utils)):
        monkeypatch.setitem(sys.modules, name, module)

    class Model:
        def __call__(self, token_batch, *, cache):
            cache[0].state = (list(token_batch[0]),)
            state['prefix_builds'].append(list(token_batch[0]))
            return [0.0]

    state['model'] = Model()
    return state


def agent_args(protocol):
    return SimpleNamespace(max_context_tokens=1_000_000, do_sample=False, temperature=0.0, top_p=1.0, top_k=0,
                           generation_protocol=protocol, tool_interface=TOOL_INTERFACE,
                           replace_text_guidance='original')


def load_variant(name='inventory'):
    fixtures = json.loads((EVIDENCE / 'fixtures.json').read_text())
    return next(item for item in fixtures['variants'] if item['variant_id'] == name)


def scripted_turn_outputs(name, arguments, tool):
    """Slot outputs for one call in the controller's schema order."""
    properties = tool['function']['parameters']['properties']
    values = []
    for key in properties:
        value = arguments[key]
        values.append(str(value) if not isinstance(value, str) else value)
    return [name, *values]


def drive(fake, protocol, horizon, tmp_path, *, tokenizer=None):
    variant = load_variant()
    all_tools = run_local_agent.tools_for('structured_replace', public_tests=True, tool_interface=TOOL_INTERFACE)
    workspace = SyntheticWorkspace(tmp_path / protocol, variant)
    tokenizer = tokenizer or CharTokenizer()
    messages = initial_messages(variant, horizon)
    lines = len(variant['source'].splitlines())
    plan = [('list_files', {}), ('read_lines', {'path': variant['path'], 'start': 1, 'end': lines}),
            ('search_source', {'query': 'total'}), ('run_public_tests', {})]
    record = []
    for turn in range(1, horizon + 1):
        tokenizer.turn = turn
        tools = tools_for_turn(all_tools, turn, horizon)
        name, arguments = plan[(turn - 1) % len(plan)] if turn < horizon else ('finish', {'summary': 'done'})
        tool = next(t for t in tools if t['function']['name'] == name)
        fake['outputs'][:] = scripted_turn_outputs(name, arguments, tool)
        before_calls, before_caches = len(fake['calls']), len(fake['caches'])
        args = agent_args(protocol)
        prompts = []
        original = run_local_agent.capture_rendered_prompt
        run_local_agent.capture_rendered_prompt = lambda a, p, _s=prompts: (_s.append(p), original(a, p))[1]
        try:
            raw, _, _ = run_local_agent.generate_schema_slot_turn(fake['model'], tokenizer, messages, tools,
                                                                  limit=256, args=args)
        finally:
            run_local_agent.capture_rendered_prompt = original
        status, call = run_local_agent.parse_tool_output_v1(raw, tools)
        assert status == 'complete_tool_call', raw
        detail = args._last_generation_detail
        expected_prefix = tokenizer.encode(tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                                         add_generation_prompt=False))
        record.append({'call': call, 'calls': fake['calls'][before_calls:], 'detail': detail,
                       'expected_prefix': expected_prefix, 'new_caches': len(fake['caches']) - before_caches,
                       'audit': audit_shared_prefix(tokenizer, messages, tools, detail, prompts)
                       if protocol == 'schema_slots_v10' else None})
        result = workspace.execute(call['name'], call['arguments'])
        messages = messages + [
            {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': call}]},
            {'role': 'tool', 'name': call['name'], 'content': json.dumps(result, ensure_ascii=False)},
        ]
    return record


@pytest.mark.parametrize('horizon', [8, 16])
def test_v10_cache_is_rebuilt_from_each_turns_changing_tool_history(fake_mlx, tmp_path, horizon):
    v9 = drive(fake_mlx, 'schema_slots_v9', horizon, tmp_path)
    v10 = drive(fake_mlx, 'schema_slots_v10', horizon, tmp_path)
    assert [t['call'] for t in v9] == [t['call'] for t in v10]
    previous_prefix = None
    for turn in v10:
        prefix = turn['expected_prefix']
        assert prefix != previous_prefix  # tool responses change the history every turn
        assert turn['detail']['shared_prefix_cache']['prefix_tokens'] == len(prefix)
        for call in turn['calls']:
            assert call['initial_cache'] == prefix
            assert call['prompt']  # the suffix is never empty
        assert all(stage['kv_cache_reused_prefix_tokens'] == len(prefix) for stage in turn['detail']['stages'])
        assert turn['audit']['violations'] == []
        assert turn['new_caches'] == 1 + len(turn['detail']['stages'])
        previous_prefix = prefix
    for turn9, turn10 in zip(v9, v10):
        for call9, call10 in zip(turn9['calls'], turn10['calls']):
            assert call9['initial_cache'] is None
            assert turn10['expected_prefix'] + call10['prompt'] == call9['prompt']


def test_tokenizer_boundary_mismatch_falls_back_without_changing_actions(fake_mlx, tmp_path):
    tokenizer = CharTokenizer(boundary_mismatch_turns={3, 5})
    v9 = drive(fake_mlx, 'schema_slots_v9', 8, tmp_path)
    v10 = drive(fake_mlx, 'schema_slots_v10', 8, tmp_path, tokenizer=tokenizer)
    assert [t['call'] for t in v9] == [t['call'] for t in v10]
    for turn_number, turn in enumerate(v10, start=1):
        stages = turn['detail']['stages']
        if turn_number in {3, 5}:
            assert all(stage['kv_cache_reused_prefix_tokens'] == 0 for stage in stages)
            assert {stage['kv_cache_fallback_reason'] for stage in stages} == {'rendered_prompt_not_exact_prefix'}
            assert all(call['initial_cache'] is None for call in turn['calls'])
        else:
            assert all(stage['kv_cache_fallback_reason'] is None for stage in stages)
        assert turn['audit']['violations'] == []


def test_v10_caches_are_released_after_each_turn(fake_mlx, tmp_path):
    drive(fake_mlx, 'schema_slots_v10', 8, tmp_path)
    gc.collect()
    assert fake_mlx['caches']
    assert all(ref() is None for ref in fake_mlx['caches'])


def test_prefix_audit_flags_forged_reuse_and_equal_length_prompts():
    tokenizer = CharTokenizer()
    messages = [{'role': 'user', 'content': 'abc'}]
    prefix = tokenizer.render(messages)
    good = prefix + '<user>x</user><assistant>'
    import hashlib
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()  # noqa: E731

    def detail(prompt, reused, prefill):
        return {'shared_prefix_cache': {'prefix_tokens': len(prefix), 'prefix_text_sha256': sha(prefix)},
                'stages': [{'prompt_sha256': sha(prompt), 'full_prompt_tokens': len(prompt),
                            'kv_cache_reused_prefix_tokens': reused, 'prefill_input_tokens': prefill}]}

    ok = audit_shared_prefix(tokenizer, messages, [], detail(good, len(prefix), len(good) - len(prefix)), [good])
    assert ok['violations'] == []
    forged = 'X' + good[1:]
    bad = audit_shared_prefix(tokenizer, messages, [], detail(forged, len(prefix), len(forged) - len(prefix)), [forged])
    assert 'stage_0_reused_non_exact_prefix' in bad['violations']
    equal = audit_shared_prefix(tokenizer, messages, [], detail(prefix, len(prefix), 0), [prefix])
    assert 'stage_0_reused_non_exact_prefix' in equal['violations']
    wrong_suffix = audit_shared_prefix(tokenizer, messages, [], detail(good, len(prefix), 3), [good])
    assert 'stage_0_suffix_length_mismatch' in wrong_suffix['violations']
    missing = audit_shared_prefix(tokenizer, messages, [], detail(good, len(prefix), len(good) - len(prefix)), [])
    assert 'stage_prompt_capture_count_mismatch' in missing['violations']


def test_interval_matches_hand_computed_t_interval_and_is_seeded():
    values = [10.0, 12.0, 14.0]
    out = interval(values, seed=7, resamples=2000)
    assert out['mean'] == pytest.approx(12.0)
    assert out['t95_low'] == pytest.approx(12.0 - 4.303 * 2.0 / 3 ** 0.5)
    assert out['t95_high'] == pytest.approx(12.0 + 4.303 * 2.0 / 3 ** 0.5)
    assert out == interval(values, seed=7, resamples=2000)
    assert 10.0 <= out['bootstrap95_low'] <= out['bootstrap95_high'] <= 14.0
    assert interval([5.0], seed=1, resamples=10) == {'n': 1, 'mean': 5.0, 'min': 5.0, 'max': 5.0}


@pytest.mark.parametrize('variant_id', ['inventory', 'textstats', 'intervals'])
def test_grader_controls_separate_baseline_fix_and_mutant(variant_id):
    variant = load_variant(variant_id)
    hidden = json.loads((EVIDENCE / 'grader' / 'hidden_tests.json').read_text())[variant_id]
    controls = json.loads((EVIDENCE / 'grader' / 'qualification_sources.json').read_text())
    assert not run_function_tests(variant['source'], variant['public_tests'])['passed']
    assert not run_function_tests(variant['source'], hidden)['passed']
    assert run_function_tests(controls['known_fix'][variant_id], variant['public_tests'])['passed']
    assert run_function_tests(controls['known_fix'][variant_id], hidden)['passed']
    assert not run_function_tests(controls['wrong_mutant'][variant_id], hidden)['passed']


def test_function_test_runner_fails_closed():
    assert run_function_tests('def f(:\n', 'def test_x():\n    assert True\n')['passed'] is False
    assert run_function_tests('x = 1\n', 'X = 1\n')['passed'] is False  # no tests collected
    looping = 'def test_spin():\n    while True:\n        pass\n'
    assert run_function_tests('x = 1\n', looping, timeout_s=3)['passed'] is False
