# Task: speed up the rollout edit-apply step

Target file: `edit_format.py` (a copy of `src/vare/learning_pipeline/edit_format.py`).
It turns a model completion into SEARCH/REPLACE blocks (`parse_blocks`) and applies
them to a source file (`apply_blocks`). The learning pipeline runs it once per rollout
before grading.

Goal: reduce the wall time of `apply_blocks(original, parse_blocks(completion))`
without changing any observable result (`ok`, `reason`, `new_text`) for any input.

Submit one unified diff that modifies only `edit_format.py`. Diffs that touch any other
path, add files, or change file modes are rejected before anything runs.

What the grader does (you cannot see or change its tests, inputs or timer):

- runs the public tests in this directory and additional held-out behaviour tests in a
  network-less sandbox; every test must pass, a skipped test counts as a failure;
- compares your outputs with the unmodified module on freshly seeded random inputs;
  any difference rejects the proposal;
- times the unmodified and the modified module on fresh inputs of about 3000
  completions against generated Python files of 800 to 2500 lines (one to three blocks
  each, about 60% of blocks differ from the file only by trailing whitespace), in a
  randomized interleaved order with a parent-side clock;
- accepts only if the 95% interval of the geometric-mean speedup lies above 1.0 and the
  point estimate is at least 1.05.

Code that inspects the test runner, the interpreter, process state, files outside the
module, or the clock is out of scope and is quarantined or rejected.
