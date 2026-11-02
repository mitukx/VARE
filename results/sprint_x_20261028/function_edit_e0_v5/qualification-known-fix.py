def normalize_batch_size(requested: int) -> int:
    """Return a batch size that is at least one."""
    return max(1, requested)
