# Coding-agent failure analysis v1

This diagnostic reads the raw outputs retained for PR #100, #101, and #102. These tasks and prompts are consumed Development data. None of these trajectories is fresh validation, training evidence, or held-out evidence. The source files and grader records remain unchanged.

## Results by experiment

| Experiment | Raw evidence | Observation |
|---|---|---|
| PR #100, Qwen3-4B single-turn patch feasibility | `results/qwen3-4b-mlsys-development-feasibility-v1/` | 15/15 rollouts failed patch format parsing on three tasks. The retained sample contains prose/fenced diff output, fabricated or non-matching hunk headers, and a patch that proposes a semantic change while also duplicating incompatible logic. No tool-mediated inspection or repair was available. Reward was constant zero.
| PR #101, stacked-video task | `results/transformers-49752-qwen-tool-feasibility-v1/` | 3 rollouts, 0 verified successes. Greedy and sample-2 made no edit and exhausted no-tool retries after reading/searching; sample-1 attempted two patches, the first did not apply and the second was corrupt. The frozen summary records 0/3, zero reward variance, and no policy update.
| PR #102, tokenizer-routing and fusion-registration tasks | `results/transformers-phase-a-structured-v1/` | 18 rollouts, 7 applied edits, 0 verified success. All 18 independent grades and all 18 rewards were failures/zero. The rollout summary records 2 consumed tasks, pass@3=0 on each, and no learning signal.

## PR #102 raw trajectory classification

Across the 18 trajectories, stop reasons were: 8 no-tool-call retries exhausted, 6 turn-budget exhausted, 3 no-tool-call after an edit, and 1 context-budget exhausted. Eleven stale-revision edit rejections were recorded; six attempts returned `edit would not change the file`. Eight runs executed public tests, all failing. These are overlapping events, not mutually exclusive counts. No trajectory made a premature verified-success claim; the final grader remained authoritative and rejected every candidate.

- **Format/parse:** PR #100's 15/15 format failures happened before application or grading.
- **Inspection/localization:** the PR #102 early versions repeatedly read large sequential regions and then stopped without an edit (six such rollouts); the controller did not ensure a targeted symbol search before source reads.
- **Semantic repair:** seven PR #102 edits applied, but none passed its public/independent checks. On Tokenizer Routing, two v3 seeds produced the same diff hash after public-test failure; v4 had repeated no-op edits while the missing override behavior remained.
- **Stale workspace:** Fusion Registration v3/v4 contains repeated `workspace revision is stale` errors. Several traces read again but still submitted an obsolete revision or ended before a valid repair.
- **Feedback use:** tests exposed failing assertions, but repair actions repeated ineffective changes or did not complete a new targeted inspection.
- **Budget:** one PR #102 rollout exhausted context; six exhausted turn budget. PR #101's failed patch application consumed its remaining turns without a valid applied edit.
- **Grader:** no infrastructure failure is needed to explain the negative outcomes: pinned calibrations passed baseline-fail/known-fix-pass, and every model-generated candidate was independently rejected.

## Controller response

The follow-up controller change requires issue-term search before reading source, caps structured source reads at 40 lines / 8,000 characters, reports complete-line `end_line` and `next_line` metadata, and explicitly asks for a new targeted action after failed public tests. Edit results remain bound to exact preimages/revisions; the finalizer still only marks a candidate ready for independent grading and never asserts verified success. Focused regression coverage is in `tests/test_run_local_agent_structured_v2.py` and `tests/test_agent_structured_edit.py`.

This change is a testable controller hypothesis. It has not yet been validated by a new model study. The two v4 exploratory rollouts on BLOOM and Phi3 are separately preserved as an interrupted, non-authoritative run in `results/transformers-coding-agent-feasibility-v4/`.
