"""Public feedback contract for the stacked-frame video regression."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from transformers import MiniCPMV4_6VideoProcessor
from transformers.video_utils import VideoMetadata


def test_stacked_visual_units_keep_aligned_shape_metadata() -> None:
    frames = np.random.default_rng(20261021).integers(
        0, 256, size=(8, 64, 64, 3), dtype=np.uint8
    )
    processor = MiniCPMV4_6VideoProcessor(
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        do_sample_frames=True,
        max_slice_nums=5,
        scale_resolution=448,
        patch_size=14,
        slice_mode=True,
        max_num_frames=20,
        stack_frames=3,
    )
    result = processor(
        frames,
        video_metadata=[VideoMetadata(total_num_frames=8, duration=4.0, fps=2.0)],
        return_tensors="pt",
    )
    control = MiniCPMV4_6VideoProcessor(
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        do_sample_frames=True,
        max_slice_nums=5,
        scale_resolution=448,
        patch_size=14,
        slice_mode=True,
        max_num_frames=20,
        stack_frames=1,
    )(
        frames,
        video_metadata=[VideoMetadata(total_num_frames=8, duration=4.0, fps=2.0)],
        return_tensors="pt",
    )
    control_sizes = control["target_sizes_videos"]
    if hasattr(control_sizes, "tolist"):
        control_sizes = control_sizes.tolist()
    main_grids = {tuple(size) for size in control_sizes}
    assert len(main_grids) == 1
    main_grid = main_grids.pop()
    frames_per_video = list(result["num_frames_per_video"])
    sizes = result["target_sizes_videos"]
    if hasattr(sizes, "tolist"):
        sizes = sizes.tolist()
    sizes = list(sizes)
    assert frames_per_video == [8]
    assert len(sizes) == sum(frames_per_video)
    assert len(result["num_patches_per_frame"]) == sum(frames_per_video)
    assert len(sizes) == 8
    grids = [tuple(size) for size in sizes]
    assert all(grid == main_grid for grid in grids[::2])
    assert all(grid != main_grid for grid in grids[1::2])
    assert len(set(grids[1::2])) == 1


if __name__ == "__main__":
    test_stacked_visual_units_keep_aligned_shape_metadata()
