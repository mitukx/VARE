# VARE Flagship Mission Status

## Current checkpoint

- Milestone: `p3-fresh-repository-task-qualification`
- Last verified remote `main`: `ce097a5d2c695d4fa7a2e2119c3e25bc3a460578`
- Current work branch: `codex/fresh-dev-candidate-discovery-v1`
- Active PR: none; candidate qualification is still underway.
- Last integration: PR #152 merged at the verified `main` SHA above. PR #150 is also merged; its repaired checks passed at exact head. PR #149 is an unrelated older open PR.

## Candidate qualification

The metadata scan retained 3,305 merged-PR rows and 1,278 metadata-review candidates. A deterministic sample selected Transformers #49418, TRL #7668, and MLX #4778. Their exact source/test inputs and test delta hashes are frozen in `results/mlsys-fresh-development-task-qualification-v1/source-test-input-manifest.json` (SHA-256 `d29cfbffc94e3f66711094f7ce72cefb0bb0580cc10b6a5a016d98c67fa60633`). All remain unqualified and unassigned.

Transformers #49418:

- Baseline attempts 1 and 2 are retained as inconclusive dependency and selector failures (zero cases).
- Attempt 3 selected six cases but failed at Quanto’s read-only lazy-extension cache before any targeted assertion; retained as an inconclusive environment failure.
- Under addendum 4 and runtime v3, baseline attempt 004 executed all six exact nodes. Four cases failed at the expected residual-length assertion and two passed. This is a valid baseline negative control.
- Two fixed-arm launch attempts are retained with zero test cases: attempt 001 had two malformed node strings, and attempt 002 passed an extra `python3` token after Docker’s `--entrypoint=python3`, so pytest did not start. Addendum 5 (SHA-256 `e16c19fb20ea09e4a559b6e3d8353dd6ebef046c3d0cf9ed7b8744847104ec86`) still freezes the exact selector list and authorizes the fixed run; the next launch will pass only the remaining `-m pytest ...` arguments.
- The exact fixed-source outcome has not yet been observed. No task qualifies yet.

Runtime v3 is pinned to `sha256:d93a01090a15c8c9ce47d7e08c6bc935f1abb83683f6318dff866a07f08f8ff2`. A runtime-only preflight compiled and loaded Quanto’s CPU extension with one build job and an executable 256 MiB tmpfs mounted only at its cache path. This is runtime preparation, not a task-test result. All attempts and hashes are under `results/mlsys-fresh-development-task-qualification-v1/`.

- Qualified fresh Development tasks: **0**.
- Coding capability: **NO-GO** pending source-level task qualification and independent model evaluation.
- SFT, policy updates, GRPO/RL comparison, Development inference, and Held-out evaluation: **NOT RUN**.
- Paid compute/API spend: **$0**; accelerator use: **none**.

## Immediate next action

Reverify the frozen source/test hashes and run the six Transformers #49418 fixed-source nodes once, deriving selectors from addendum 5. Because Docker already sets `--entrypoint=python3`, pass only `-m pytest ...` from the frozen command string. Qualification requires all six pass with zero skips. Then qualify TRL #7668 and MLX #4778 sequentially under addendum 1. Do not assign tasks to Development unless all qualification gates pass.

## External blockers

- No fresh task is qualified yet; public pretraining exposure of the selected upstream patches remains unknown.
- The prior historical queue has no fresh Development task. New candidates remain unassigned until all source-level gates pass.
- No post-update policy result or independent fresh-task model evaluation has been authorized or executed.
