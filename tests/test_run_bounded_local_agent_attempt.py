from unittest.mock import patch

import pytest

from scripts.run_bounded_local_agent_attempt import process_tree_rss, run_limited


def test_process_tree_rss_sums_root_and_nested_descendants() -> None:
    ps_output = "10 1 100\n11 10 50\n12 1 200\n13 11 25\n"
    with patch(
        "scripts.run_bounded_local_agent_attempt.subprocess.check_output",
        return_value=ps_output,
    ):
        assert process_tree_rss(10) == (100 + 50 + 25) * 1024


def test_run_limited_records_command_and_output(tmp_path) -> None:
    record = run_limited(
        ["/usr/bin/true"],
        wall_seconds=2,
        rss_limit_bytes=1024**3,
        log_path=tmp_path / "attempt.log",
    )

    assert record["exit_code"] == 0
    assert record["limit_reason"] is None
    assert record["command"] == ["/usr/bin/true"]
    assert record["log_bytes"] == 0


def test_run_limited_stops_process_when_rss_budget_is_exceeded(tmp_path) -> None:
    record = run_limited(
        ["/bin/sleep", "5"],
        wall_seconds=3,
        rss_limit_bytes=1,
        sample_seconds=0.05,
        log_path=tmp_path / "attempt.log",
    )

    assert record["limit_reason"] == "process_tree_rss_limit_exceeded"
    assert record["max_observed_process_tree_rss_bytes"] > 1


def test_run_limited_stops_process_at_wall_budget(tmp_path) -> None:
    record = run_limited(
        ["/bin/sleep", "5"],
        wall_seconds=1,
        rss_limit_bytes=1024**3,
        sample_seconds=0.05,
        log_path=tmp_path / "attempt.log",
    )

    assert record["limit_reason"] == "wall_clock_limit_exceeded"
    assert record["elapsed_seconds"] < 3


@pytest.mark.parametrize(
    ("wall_seconds", "rss_limit_bytes", "sample_seconds"),
    [(0, 1, 0.5), (1, 0, 0.5), (1, 1, 0)],
)
def test_run_limited_rejects_invalid_budget(
    tmp_path, wall_seconds, rss_limit_bytes, sample_seconds,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_limited(
            ["/usr/bin/true"],
            wall_seconds=wall_seconds,
            rss_limit_bytes=rss_limit_bytes,
            sample_seconds=sample_seconds,
            log_path=tmp_path / "attempt.log",
        )
