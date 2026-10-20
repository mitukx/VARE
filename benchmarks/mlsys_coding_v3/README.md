# MLSys coding pack v3 — small exploratory cohort

This frozen pack combines five previously used v1 development tasks with two newly assigned tasks from the unassigned v2 author pool: EMA bias correction and label smoothing. The two v3 tasks were not found in prior run evidence; they are exploratory held-out tasks only. They are authored single-file utilities, not upstream issue patches. The cohort is too small to support broad MLSys generalization claims.

The GQA repeat candidate from v2 is excluded because its grader checks input mutation in addition to return values, while the current isolated call schema only records return values. It will not be scored until that contract is represented in the independent grader.

Both models use one deterministic greedy run per task; the run is a matched pilot, not a stochastic sample or scale-only comparison.
