# Task: Keep FPS sampling nonempty for a short nonempty video

Fix `default_sample_indices_fn` in `video_utils.py`. When the source has at least one frame and a positive target FPS, FPS-based selection must retain at least one frame even when truncation would otherwise calculate zero. Preserve the existing behavior for ordinary sampling, empty sources, fixed `num_frames`, and missing source FPS.
