# ResearchGym v1

CPU task pack for `vare research`. It is separate from the frozen coding-RL
readiness protocols and from held-out splits used by earlier model studies.

| Task | Type | What the hidden grader checks |
| --- | --- | --- |
| `grad_accum_scale_v1` | training correctness | Mean of microbatch gradients on hidden seeds, plus `ValueError` on empty input |
| `loop_invariant_score_v1` | performance | Value match and a 2x speedup against the untouched baseline, with an A/A noise band |
| `descent_sign_debug_v1` | experiment debug | Held-out affine fit and one mean-reduced gradient step; reported loss is ignored |

`controls/known_good` and `controls/incorrect` are grader calibration files.
They are not copied into an agent workspace. A known-good pass is not an
agent result.

Hidden seeds and acceptance thresholds live in `src/vare/research_lab/evaluation.py`.
Task JSON only carries the public seed. The worker process does not receive
expected values and does not get the repository on `PYTHONPATH`.

This suite does not establish model capability, GPU scaling, or permission to train.
