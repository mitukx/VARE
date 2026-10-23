# Agent controller follow-up v1

The retained greedy and sampled traces both sent a `path` field to `list_files`; the locked runner rejected both calls because the read-only tool schema accepted no arguments. The greedy trace later applied a patch that changed the Apache license header, and the task grader correctly rejected that candidate. These are concrete controller-safety and tool-compatibility defects from the completed Development run.

The follow-up changes the read-only `list_files` schema to accept an optional path filter and still returns an error for paths outside the exact allowlist. It also compares copyright/license comment lines before and after every applied edit. If an edit changes those lines, the runner restores all affected files and records a rejected edit. Ordinary source edits with unchanged headers remain allowed.

Focused regressions pass **23/23** tests. The related grading and evidence suites pass **70 tests with 4 skips**; Ruff and `git diff --check` pass. The new tests cover the recorded extra-argument shape, valid and invalid path filters, rejection with exact source-byte restoration after a license-header mutation, and acceptance of an ordinary code change that preserves the header.

This is a source-level correction informed by Development traces. It does not alter the frozen v1 experiment, and no inference or grading was repeated on the consumed task. Any future model run uses a new protocol and a different eligible task.
