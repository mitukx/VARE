# Independent AI review record

Reviewer: a separate AI agent, read-only with respect to this repository.
This is not human review, external maintainer acceptance, or external merge.

## Executed during development

Using the real RVL HFLocalBackend and random tiny GPT-2, the reviewer reproduced
active evaluation/rollout mode and CPU-RNG contamination, including scorer
failure after generation, and cancelled-worker overlap on the baseline.
It then executed the proposed inference helper with:

- evaluate/rollout x worker success/error, three cancellations each;
- populated AdamW moments, complete trainer state comparison;
- a one-shot CPU RNG restore exception, followed by next-call recovery;
- real generation followed by scorer failure.

The output is `ai-review-probes.json`. These scratch probes used RVL
`7568c6e39015f18bec337174bacf39caac4f2477` (PR #92), not the main revision
used by the retained canonical replay. Their original scripts are retained in
the local work record and use host-specific paths; they are supplementary
development evidence. The maintained tests and portable reproducer cover the
corresponding contract without those path dependencies.

## Final static review (no test re-execution)

The reviewer read implementation commit
`3139f4dc7cf4421e197af89c0010ecbad81e4d75`, the one-entry script and HF probe,
verified every source/artifact hash in `manifest.json`, checked the 14-case XML
counts (12/10/0 failures), fixed HF state record, and 338-pass full-suite log.
It found no new actionable defect within the stated CPU contract. The adapter
source SHA256 was
`97988fb5b7620b7365920a87b07152f311940ae5b17d90929d8002b5e7562aea`.

## Limits and separate finding

CPU and initialized-CUDA RNG is the implemented scope; only CPU was exercised.
MPS/XPU, arbitrary scorer state, independent global RNG users, detached backend
threads, and hard termination deadlines are unverified or unsupported here.

The reviewer also executed a generic engine `LoopHooks` fixture where an
incumbent evaluation error leaves its sibling evaluation live during candidate
discard. This engine-level task-ownership gap is documented as open; it is not
silently included in the adapter fix or its passing claims.
