# TRL async stale-drop resume cursor v1 — superseded

> **Protocol correction (2026-10-18):** v1 compared dataset row IDs without
> epoch identity. When a resume cursor was near the dataset end, the async
> rollout loop legitimately wrapped to row 2 in the next epoch, and the
> candidate's “not replayed immediately” assertion treated that new occurrence
> as the old event. The later v1 replay retains this false-negative test result;
> the earlier all-pass v1 bundle is not a reliable epoch-invariant replay
> measure. Use the corrected [v2 report](trl-async-stale-resume-v2-report.md)
> and its absolute stream-position protocol for the current result.

## Finding

TRL issue [#7274](https://github.com/huggingface/trl/issues/7274) describes a checkpoint cursor that stops at the first group rejected by the rollout staleness filter. Later groups can already have trained, but resume starts again at that stale group and regenerates the later rows. The same cursor calculation exists in AsyncDistillation.

We reproduced the failure against TRL `6d6901c5e3963108076131ac4979d344566b13a2` and prepared a narrow candidate fix at `a85900829d86bcd45b7db0310725f28d3152289b`. The patch records stale-dropped IDs as resolved for cursor calculation, keeps actual trained counts separate for epoch accounting, and covers both AsyncGRPO and AsyncDistillation. Its source patch is retained at [`upstream-fix.patch`](../contributions/trl-async-stale-resume-v1/upstream-fix.patch).

## Reproduction

The pinned public issue reproduction runs the production rollout scheduling, staleness filter, checkpoint save and resume path. It stubs only `_generate_one` and `_score_group`; no model weights, optimizer, vLLM, GPU, or independent task evaluator are involved. Three repetitions per arm were run on an Apple M1 Pro with 32 GiB RAM, using CPU, Python 3.12.12, PyTorch 2.9.1, Transformers 5.20.1, Datasets 5.1.0, Accelerate 1.12.0, and pytest 8.4.2. The dependency pins are in [`requirements/trl-async-stale-resume-v1.txt`](../requirements/trl-async-stale-resume-v1.txt), and the frozen source/protocol record is [`protocols/trl_async_stale_resume_v1.lock.json`](../protocols/trl_async_stale_resume_v1.lock.json).

| Arm | Saved cursor | Resume row | Trained rows repeated after resume | Outcome |
| --- | ---: | ---: | ---: | --- |
| Baseline, locked tail-tolerant repeats (n=3) | 2 | 2 | 145 each | 3/3 reproduced the stale-hole replay |
| Candidate (n=3) | 147–151 | 148 each | 1 each | 3/3 resumed past the stale group and matched trained-count state |

The candidate reduced repeated rows from 145 to 1 in this fixture. The single repeated row is an asynchronous in-flight tail; this experiment does not establish exactly-once execution. The first exploratory set is also retained: its unmodified public assertion required at least 145 repeated rows, so two baseline repetitions that observed 144 rows exited nonzero even though each saved and resumed at group 2. The follow-up lock uses the same ten-row tail allowance already used by the reproduction's overlap check. These earlier failures are kept in [`run-1-original-threshold`](../results/trl-async-stale-resume-v1/run-1-original-threshold/), separate from the locked six-run bundle.

The candidate's focused upstream tests passed: 9/9 tests in `test_async_stale_resume.py`. VARE's evidence parser and tamper-detection tests pass as well. The VARE runner downloads the pinned issue Gist, verifies its SHA-256, applies the retained source deltas, checks both TRL revisions and dependency versions, runs all six reproductions, and writes raw logs plus a hashed manifest. Re-run it with:

```bash
python -m venv --system-site-packages /tmp/vare-trl-env
/tmp/vare-trl-env/bin/python -m pip install -r requirements/trl-async-stale-resume-v1.txt
git clone https://github.com/huggingface/trl.git /tmp/trl-source
git -C /tmp/trl-source fetch origin 6d6901c5e3963108076131ac4979d344566b13a2
git -C /tmp/trl-source worktree add --detach /tmp/trl-base 6d6901c5e3963108076131ac4979d344566b13a2
git -C /tmp/trl-source remote add vare https://github.com/mitukx/trl.git
git -C /tmp/trl-source fetch vare codex/async-stale-resume-cursor
git -C /tmp/trl-source worktree add --detach /tmp/trl-candidate a85900829d86bcd45b7db0310725f28d3152289b
python scripts/reproduce_trl_async_stale_resume_v1.py \
  --python /tmp/vare-trl-env/bin/python \
  --baseline-source /tmp/trl-base \
  --candidate-source /tmp/trl-candidate \
  --output /tmp/trl-async-stale-resume-replay
python scripts/audit_trl_async_stale_resume_v1.py /tmp/trl-async-stale-resume-replay
```

The repository-native patch and regression tests are retained in the upstream patch file. The issue reproduction and API adaptations are pinned and hash-checked; no full copy of the reporter's Gist is stored.

## Upstream status and limits

Upstream PR [#7786](https://github.com/huggingface/trl/pull/7786) was automatically closed because it lacked the required first-contributor checklist and linked issue assignment. Issue #7274 remains open and unassigned. No upstream code review or merge occurred; this report records a tested candidate contribution, not an upstream fix.

This establishes a source-level resume-cursor correctness defect and a tested candidate correction. It does not measure optimizer updates, checkpointed optimizer/RNG consistency, task success, model capability, throughput, or distributed-scale behavior. The only new observation is reduced duplicate training rows in the bounded CPU reproducer.
