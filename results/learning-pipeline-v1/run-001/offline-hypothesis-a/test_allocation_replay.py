"""Synthetic-fixture checks (E0) for allocation_replay.py: budget, cap, and label blindness."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys

_spec = importlib.util.spec_from_file_location("allocation_replay", Path(__file__).with_name("allocation_replay.py"))
assert _spec and _spec.loader
ar = importlib.util.module_from_spec(_spec)
sys.modules["allocation_replay"] = ar
_spec.loader.exec_module(ar)


def _units(flip: bool = False) -> dict[str, list]:
    out = {}
    for t in range(4):
        us = []
        for k in range(3):
            passed = (t + k) % 2 == 0
            us.append(ar.Unit(task=f"t{t}", key=f"c{t}{k}", first_index=k, agreement=1 + k % 2, diff_lines=k,
                              stop=k != 2, tokens=10 * k, passed=passed != flip, output_shas=(f"o{t}{k}",)))
        out[f"t{t}"] = us
    return out


def test_budget_respected_and_cap_applied() -> None:
    units = _units()
    for pol in [*ar.baselines("short").values(), ar.make_selective("short", "exploit")]:
        for b in (1, 3, 7, 12):
            g = pol(units, b, random.Random(0))
            assert len(g) <= b
            assert len({(u.task, u.key) for u in g}) == len(g)
    all_units = [u for us in units.values() for u in us]
    assert ar.yield_of(all_units)["rft"] == sum(min(2, sum(u.passed for u in us)) for us in units.values())


def test_non_adaptive_policies_ignore_hidden_labels() -> None:
    a, b = _units(False), _units(True)
    for name, pol in ar.baselines("agreement").items():
        ka = [(u.task, u.key) for u in pol(a, 6, random.Random(3))]
        kb = [(u.task, u.key) for u in pol(b, 6, random.Random(3))]
        assert ka == kb, name


def test_selective_first_round_is_label_blind_and_stops_at_cap() -> None:
    a, b = _units(False), _units(True)
    pa = ar.make_selective("short", "exploit")(a, 4, random.Random(1))
    pb = ar.make_selective("short", "exploit")(b, 4, random.Random(1))
    assert [(u.task, u.key) for u in pa] == [(u.task, u.key) for u in pb]
    rich = {"t": [replace(u, task="t", passed=True) for u in _units()["t0"]]}
    assert len(ar.make_selective("short", "exploit")(rich, 3, random.Random(0))) == 2
