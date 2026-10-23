"""Candidate-side request handler; trusted expected values stay in VARE."""

from __future__ import annotations

from typing import Any


def run(case: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from transformers import MiniCPMV4_6VideoProcessor
    from transformers.video_utils import VideoMetadata

    seed = int(case["seed"])
    frames = np.random.default_rng(seed).integers(
        0, 256, size=(8, 64, 64, 3), dtype=np.uint8
    )
    kwargs = {
        "do_resize": True,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "do_convert_rgb": True,
        "do_sample_frames": True,
        "max_slice_nums": 5,
        "scale_resolution": 448,
        "patch_size": 14,
        "slice_mode": True,
        "max_num_frames": 20,
    }
    metadata = [VideoMetadata(total_num_frames=8, duration=4.0, fps=2.0)]
    baseline = MiniCPMV4_6VideoProcessor(**kwargs, stack_frames=1)(
        frames, video_metadata=metadata, return_tensors="pt"
    )
    main_grids = {tuple(size) for size in baseline["target_sizes_videos"].tolist()}
    if len(main_grids) != 1:
        raise ValueError("control frames did not produce one main-frame grid")
    main_grid = list(main_grids.pop())
    output = MiniCPMV4_6VideoProcessor(**kwargs, stack_frames=3)(
        frames, video_metadata=metadata, return_tensors="pt"
    )
    return {
        "num_frames_per_video": output["num_frames_per_video"],
        "target_sizes_videos": output["target_sizes_videos"].tolist(),
        "num_patches_per_frame_count": len(output["num_patches_per_frame"]),
        "main_grid": main_grid,
    }
