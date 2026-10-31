@pytest.mark.parametrize(
    ("overrides", "feature"),
    [
        (
            {"attention_config": AttentionConfig(hisparse_config=HiSparseConfig())},
            "HiSparse",
        ),
        ({"watermark_config": WatermarkConfig(key=42)}, "watermarking"),
    ],
)
def test_v1_model_runner_rejects_v2_only_config(overrides, feature):
    config = SimpleNamespace(
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig.default_factory(async_scheduling=False),
        attention_config=AttentionConfig(),
        watermark_config=None,
        speculative_config=None,
        model_config=None,
    )
    config._dflash_needs_multi_kv_group = lambda: False
    config._is_dflash_candidate_draft = lambda: False
    assert VllmConfig._get_v1_model_runner_unsupported_features(config) == []

    for name, value in overrides.items():
        setattr(config, name, value)
    assert VllmConfig._get_v1_model_runner_unsupported_features(config) == [feature]
