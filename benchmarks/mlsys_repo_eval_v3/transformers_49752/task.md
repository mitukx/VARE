# Task: fix stacked-frame grouping in the MiniCPMV4_6 video processor

When `stack_frames` is greater than one, the video processor must preserve a
one-to-one relationship between its visual units and the per-unit target sizes
and patch counts. A stack of three sampled frames contains alternating main
frames and composite canvases; processing must use those visual units for all
subsequent shape-grouped work. Keep the change in
`src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py`.

The candidate workspace contains the pinned `src/transformers` package and
this task statement. Do not use Git history, network access, external files,
model checkpoints, or tests other than the task-provided public smoke test.
