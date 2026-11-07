# ResearchGym v1

CPU task pack for `vare research`. It is separate from the frozen coding-RL
readiness protocols and from held-out splits used by earlier model studies.

| Task | Type | What the hidden grader checks |
| --- | --- | --- |
| `grad_accum_scale_v1` | training correctness | Mean of microbatch gradients on hidden seeds, plus `ValueError` on empty input |
| `loop_invariant_score_v1` | performance | Value match, and a paired 95% bound on time saved that exceeds the A/A noise bound |
| `descent_sign_debug_v1` | experiment debug | Held-out affine fit and one mean-reduced gradient step; reported loss is ignored |

`controls/known_good` and `controls/incorrect` are grader calibration files.
They are not copied into an agent workspace. A known-good pass is not an
agent result.

Hidden seeds and acceptance thresholds live in `src/vare/research_lab/evaluation.py`.
Task JSON only carries the public seed. The worker process does not receive
expected values and does not get the repository on `PYTHONPATH`.

Hidden grading runs only after the bubblewrap probe in
`vare.research_lab.isolation` is denied. That probe is not a trusted security
sandbox. An unavailable or unverified probe fails closed and does not fall
back to a same-user process group. `run_bounded` is that process group: no
hard address-space cap and no network namespace. A passing grade does not
mean a trusted sandbox enforced those limits.

This suite does not establish model capability, GPU scaling, or permission to train.
Measurements retained with this pack were taken on a CPU guest, not on Apple
Silicon, and `counts_as_autonomous_model_success` stays false.
