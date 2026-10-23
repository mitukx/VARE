# Phase A v6 run-2 public-test classification correction v2

The v1 correction artifact omitted one serialized invocation. The retained `transformers-49515-rope-setter-v1/greedy-0/agent.json` has both an automatic post-edit public-test result (Docker exit 125, 0.098 s) and a later explicit `run_public_tests` result (exit 125, 0.046 s). Together with the other five explicit results, the raw trajectory contains **seven serialized container invocations**: one automatic and six explicit.

The v2 append-only artifact at [`public-test-classification-correction-v2`](../results/phase-a-v6/protocol-v1/run-2/public-test-classification-correction-v2/) lists every invocation, identifies its exact location in its `agent.json`, hashes the source file and result object, and binds the correction to the original run manifest and report. All seven returned exit 125 because Colima could not see the bind source path. No test function ran; these are infrastructure launch failures, not candidate assertion failures or timeouts.

The prior v1 correction and the frozen run report, raw trajectories, and checksums remain unchanged. This correction changes no independent grader outcome: verified successes remain 0/12, and the learning decision remains NO-GO. No inference or grading was rerun.
