"""Public CPU behavior check for stacked MiniCPM-V visual units."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SOURCE = Path(__file__).parent / "src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py"


class StopAtStageTwo(Exception):
    pass


def load_preprocess(grouping_function):
    module = ast.parse(SOURCE.read_text(encoding="utf-8"))
    processor = next(
        node for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniCPMV4_6VideoProcessor"
    )
    method = next(
        node for node in processor.body
        if isinstance(node, ast.FunctionDef) and node.name == "_preprocess"
    )
    probe = ast.ClassDef(name="Probe", bases=[], keywords=[], decorator_list=[], body=[method])
    namespace = {
        "math": __import__("math"), "np": np, "torch": torch,
        "group_videos_by_shape": grouping_function,
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )
    code = ast.fix_missing_locations(ast.Module(body=[future, probe], type_ignores=[]))
    exec(compile(code, str(SOURCE), "exec"), namespace)
    return namespace["Probe"]


def test_stage_two_receives_ordered_single_frame_and_composite_units():
    observed = []

    def group_videos_by_shape(value):
        observed.extend(value)
        raise StopAtStageTwo

    Probe = load_preprocess(group_videos_by_shape)
    processor = Probe()
    processor.concat_frames_as_image = lambda subframes: torch.full(
        (1, 1, 2, 2), 100 + float(subframes[0, 0, 0, 0]), dtype=torch.float32,
    )
    video = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)

    try:
        processor._preprocess(
            [video], False, None, False, 1.0, False, None, None, 9, 448, 14, False,
            2, 4, [SimpleNamespace(duration=2.0)], None,
        )
    except StopAtStageTwo:
        pass
    else:
        raise AssertionError("probe did not reach Stage 2")

    assert len(observed) == 4, f"expected four interleaved units, got {len(observed)}"
    assert all(tuple(frame.shape) == (1, 1, 2, 2) for frame in observed)
    first_values = [float(frame[0, 0, 0, 0]) for frame in observed]
    assert first_values == [0.0, 108.0, 4.0, 112.0], first_values


if __name__ == "__main__":
    test_stage_two_receives_ordered_single_frame_and_composite_units()
