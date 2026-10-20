"""Smoke check. This file is not the grader."""

from aging_scheduler import schedule


def test_higher_base_priority_runs_first():
    jobs = [
        {"id": "a", "release": 0, "priority": 1, "service": 2},
        {"id": "b", "release": 0, "priority": 5, "service": 2},
    ]
    assert [item["id"] for item in schedule(jobs)] == ["b", "a"]


if __name__ == "__main__":
    test_higher_base_priority_runs_first()
