from accelerate.data_loader import DataLoaderShard, DataLoaderDispatcher, skip_first_batches
def test_preserve_configuration():
    for cls in (DataLoaderShard, DataLoaderDispatcher):
        loader=cls(list(range(16)), batch_size=4, _drop_last=True, _non_blocking=True)
        result=skip_first_batches(loader, num_batches=1)
        assert result._drop_last is True
        assert result._non_blocking is True
