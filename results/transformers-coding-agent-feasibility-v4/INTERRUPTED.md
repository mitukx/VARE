# Feasibility v4 execution status — interrupted, non-authoritative

Two of the eight preregistered rollouts were started: BLOOM greedy (seed 2101) and Phi3 greedy (seed 2101). Both terminated with `turn_budget_exhausted` after six tool turns, made no edit, and generated 253 / 258 tokens respectively. The trajectories show the agent spending its tool budget reading sequential source chunks without searching for the implicated implementation. Both are retained as controller-diagnostic observations.

The v4 study was stopped before the remaining six rollouts after audit found the parent grader did not forward protocol-v9's declared `pids=32` ceiling; protocol-v9 used the API default of 64. Therefore this partial run is not an authoritative completion of the frozen study. The BLOOM grading attempt used the defective v9 grader and is retained but is not counted. Phi3 was not independently graded. No task-level pass@4, success-rate, or reward-variance estimate is reported.

A corrected, versioned protocol-v10 calibration now forwards and records limits, and v5 of the agent protocol will freeze the targeted-search controller before any additional inference. The Development tasks remain Development tasks and are not held-out evidence.
