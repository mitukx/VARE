# v1 preflight failure

The predeclared reproducer imported and began iterating the pinned production `RolloutQueueDataset`, but stopped when the first stale sample triggered Accelerate's logger before `PartialState()` or `Accelerator()` had been initialized:

```text
RuntimeError: You must initialize the accelerate state by calling either `PartialState()` or `Accelerator()` before using the logging utility.
```

This is a fixture setup failure, not an observation about group admission. No v1 acceptance criterion was evaluated. The v1 source/protocol remain unchanged. v2 adds the required Accelerate process-state initialization before constructing either dataset iterator.
