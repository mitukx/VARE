# GitHub state erratum — 2026-10-28

This append-only correction applies to the PR-count sentence in `REPORT.md`. The live open-PR listing contains 18 open PRs and **9 Drafts**, not 8. PR #114 is Draft and was omitted from the earlier hand count. No research result or task eligibility decision changes.

State after the Sprint X evidence push:

- `main`: `f3ab90a36e8724441b6b9555255f8f75a0a4d529` (unchanged).
- PR #139: head `18afc60910ceb3ef5572fb2c7eadeff551bf9510`, Draft, stacked on #137; all visible checks pass.
- PR #142: head `d95dbc2704b30527a7ae95858ccce490da3b38b5`, Draft, `REVIEW_REQUIRED`; checks triggered by the evidence push were still pending at this snapshot. Its report/body was updated after the push without changing source artifacts.
- PR #140 at `34f972eb2a300711f36fd81a72b6948528dfebad` and PR #143 at `6266ec0a10920ae9ef95ccb528145ef14a452e39`: all visible CI checks pass; both remain `REVIEW_REQUIRED` and `BLOCKED` by branch protection.
- The repository requires one approving review. The live collaborator listing contains only the repository owner, so no independent collaborator review request was available. No approval or merge was fabricated.
- No PR merged; mainline SHA did not change.
