import pytest

from scripts.preflight_local_model_attempt import (
    available_memory_from_vm_stat,
    memory_pressure_free_percent,
    parse_swap_usage,
)


def test_available_memory_uses_free_and_inactive_pages() -> None:
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\n"

    assert available_memory_from_vm_stat(text) == 300 * 16384


def test_memory_pressure_parser_requires_percentage() -> None:
    assert memory_pressure_free_percent("System-wide memory free percentage: 58%\n") == 58
    with pytest.raises(ValueError, match="free percentage"):
        memory_pressure_free_percent("memory pressure unavailable")


def test_swap_usage_parses_all_units() -> None:
    parsed = parse_swap_usage("vm.swapusage: total = 3.00G used = 512.00M free = 1024.00M")

    assert parsed == {
        "total_bytes": 3 * 1024**3,
        "used_bytes": 512 * 1024**2,
        "free_bytes": 1024 * 1024**2,
    }


def test_swap_usage_fails_closed_when_a_field_is_missing() -> None:
    with pytest.raises(ValueError, match="free"):
        parse_swap_usage("total = 3.00G used = 512.00M")
