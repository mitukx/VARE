# Pydantic #13825 task qualification — attempt 003

This task passed the frozen three-arm qualification and is now a **prospective Train candidate only**. Eight upstream tests were run offline against exact baseline and fixed revisions with the same tests and harness in every arm. The baseline failed five tests, the targeted mutation failed one, and fixed source passed all eight; no arm timed out or was OOM-killed.

The run used Python 3.12 / Linux ARM64, pytest 8.4.2, and the exact `pydantic-core` 2.49.0 required by the candidate revision. The selected source and test revisions, environment and wheel hashes, arm tree hashes, selectors, outcomes, and raw qualification logs are included. Earlier scope, runner, and environment preflight failures are summarized with their SHA-256 references in `inputs/rejected-preflights.json`; the complete internal records are preserved outside this repository bundle.

This does not authorize model rollout. The frozen NO-GO gate remains unchanged at 0/15 eligible held-out tasks and explicitly forbids training or new agent rollouts while it fails. The candidate has not been assigned, exposed to a model, or counted as held-out. No held-out or Development task was accessed. No optimizer update, checkpoint, reload, or policy evaluation occurred.
