import time

from ray.data._internal.stats import DatasetStats, OperatorStatsSummary
from ray.data.block import BlockExecStats, BlockStats


def _mock_block_stats(start_time, end_time, *, num_rows=100, task_idx=None, node_id="node", unix_start=None):
    wall_time_s = end_time - start_time
    exec_stats = BlockExecStats(
        start_time_s=start_time,
        end_time_s=end_time,
        wall_time_s=wall_time_s,
        cpu_time_s=wall_time_s,
        task_idx=task_idx,
        node_id=node_id,
    )
    # The baseline dataclass predates this field. Dynamic assignment keeps the
    # exact same test input runnable against baseline and fixed source.
    object.__setattr__(
        exec_stats,
        "start_unix_time_s",
        start_time if unix_start is None else unix_start,
    )
    return BlockStats(num_rows=num_rows, size_bytes=None, exec_stats=exec_stats)


def test_dataset_throughput_calculation():
    blocks_stats = [
        _mock_block_stats(0.0, 2.0),
        _mock_block_stats(0.5, 2.5),
        _mock_block_stats(1.0, 3.0),
    ]
    summary = DatasetStats(metadata={"Map": blocks_stats}, parent=None).to_summary()
    assert summary.num_rows_per_s == 100


def test_operator_throughput_calculation():
    blocks_stats = [
        _mock_block_stats(0.0, 2.0, task_idx=1),
        _mock_block_stats(0.0, 2.0, task_idx=2),
    ]
    summary = OperatorStatsSummary.from_block_metadata(
        operator_name="MockOperator", block_stats=blocks_stats, is_sub_operator=False
    )
    assert summary.num_rows_per_s == 200 / 2.0
    assert summary.num_rows_per_task_s == 200 / (2.0 + 2.0)


def test_operator_wall_time_multi_node():
    blocks = [
        _mock_block_stats(10800.0, 10810.0, node_id="A", unix_start=1000.0, task_idx=0),
        _mock_block_stats(120.0, 130.0, node_id="B", unix_start=1002.0, task_idx=0),
    ]
    summary = OperatorStatsSummary.from_block_metadata(
        operator_name="MockOperator", block_stats=blocks, is_sub_operator=False
    )
    assert summary.time_total_s == 12.0
    assert summary.earliest_start_time == 1000.0
    assert summary.latest_end_time == 1012.0
    assert summary.num_rows_per_s == 200 / 12.0


def test_block_exec_stats_builder_records_unix_anchor():
    before = time.time()
    stats = BlockExecStats.builder().build(node_id="node")
    after = time.time()
    anchor = getattr(stats, "start_unix_time_s", None)
    assert anchor is not None
    assert before <= anchor <= after
    assert stats.wall_time_s >= 0
