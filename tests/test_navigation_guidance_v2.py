from scripts.run_local_agent import navigation_guidance_sentence


def test_v1_guidance_is_available_and_preserves_newline_contract():
    guidance = navigation_guidance_sentence('v1')
    assert 'general terms from the task' in guidance
    assert 'replacement text must also end with a newline' in guidance


def test_v2_guidance_targets_observed_search_and_repeated_read_failures():
    guidance = navigation_guidance_sentence('v2')
    assert 'one short, distinctive identifier or term' in guidance
    assert 'retry immediately with a shorter identifier' in guidance
    assert 'do not repeat an identical range' in guidance
    assert 'advance toward the hit' in guidance
    assert 'read a contiguous range that contains every line you intend to replace' in guidance
    assert 'emit real line breaks' in guidance
    assert 'backslash followed by n is two literal characters' in guidance
    assert 'replacement text must also end with a newline' in guidance
    assert 'Do not invent, reuse, or modify a lease' in guidance

