# Public repository content sanitization

## Private positioning language

The source tree at `6373d8d91ac76d5f3ba1bae53cdee002b59db16c` contained unrelated positioning language in two local engineering records. The reports now state the technical evidence limit directly: they are local engineering results and did not evaluate model capability or external task performance. The corresponding validation field now uses `agent_capability_evaluation`.

Technical citations remain where they identify prior work or an implementation relevant to the experiment. The current mainline text audit found no personal nontechnical goal statements.

The open-PR metadata search found one branded document filename linked from PR #139. PR #184 renamed that file to `docs/vare-flagship-research-sprint-20261027.md` without changing its contents, and the #139 description now points to the new path. PR #139 remains Draft; its dependent #146 remains open on the same branch. The concurrent PRs #181, #182, and #183 were left untouched.

## Run-record path redaction

The source tree contained eleven committed run records with absolute machine-specific workspace roots in recorded paths. Seven additional committed test logs contained the same kind of root. This change publishes redacted derivatives for 18 files, replacing 52 occurrences while preserving measurements, outcomes, test results, protocol definitions, and source snapshots.

For each derivative, [`results/publication-redaction-v1/manifest.json`](../results/publication-redaction-v1/manifest.json) records the source Git blob ID and SHA-256, replacement SHA-256, and number of replacements. Eleven associated evidence manifests are updated to hash the published records. Frozen protocol locks are unchanged.

The original blobs remain reachable through the source commit in Git history. No history rewrite or deletion was performed, so this change cleans the current tree without claiming erasure from earlier commits, forks, caches, or copies.

The original wording and unredacted run paths remain visible in historical commit `6373d8d91ac76d5f3ba1bae53cdee002b59db16c`, which is the parent of this cleanup. Removing those historical objects would require a separately authorized history rewrite and coordination with forks and clones.
