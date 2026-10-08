# Next study decision — 2026-10-09

## Decision

Make a generated multi-turn code/engineering repair task the next primary candidate. First run a frozen, no-update feasibility screen on the locally cached Qwen2.5-0.5B-Instruct model. Use 32 small generated mini-repository episodes across four bug families and eight bug templates (four generated variants per template), with bounded file and visible-test tools. Keep hidden tests outside the model-accessible workspace. Measure episode success, tool-call validity, CPU throughput, peak memory, and whether there is enough room to improve. Do not start training unless the task and resource gates pass.

The research question after a successful screen is:

> Under the same task generator and rollout/update budget, does verifier-reward policy optimization improve success on unseen task compositions beyond both the frozen policy and matched successful-trajectory SFT, without increasing invalid or unsafe tool actions?

The first learning study should use one policy-optimization method, one matched SFT control, multiple seeds, and an untouched composition-held-out confirmation set. A separate verifier implementation must score final environment state. Report episode success as the primary outcome; also report invalid calls, task-step count, KL, generated tokens, CPU time, RSS, and seed-level results. A reward increase without an oracle-scored success increase is a non-pass.

## Why this path

The repository already contains a bounded local-agent runner, a simulator/evaluator framework, and substantial replay, rollback, and evidence-audit work. A generated repair pack can reuse those strengths while testing a complete post-training chain: task generation, tool-use rollouts, executable reward, policy update, and held-out task outcome. It needs no remote service, paid API, or GPU. It also has a closer connection to multi-step engineering-agent behavior than a generic business-API simulator.

This choice changes the evidence target from preference-score movement to success in an executable task. It does not change the project away from post-training and reinforcement learning. Existing preference studies and trainer-integrity work remain supporting evidence; repeated nearby HH, GSM8K, BoolQ, or SNLI variants are not the next primary study.

## Alternatives reviewed

| Candidate | Value | Main limitation | Decision |
| --- | --- | --- | --- |
| Multi-turn generated code/engineering repair | Directly connects tool-using policy updates and verifier reward to a measurable software task; can reuse the local agent runner and deterministic graders. | Synthetic tasks are narrow. The model failed three runs on one harder historical repair task ([pilot record](local-agent-pilot.md)), so useful base competence is unproven; CPU rollout speed may prevent training. | **Primary feasibility screen on a new, lower-complexity generated task pack. Retire if base competence or resource gates fail.** |
| Math tool-integrated reasoning (TIR) | Exact arithmetic oracle and cached Qwen2.5-Math-1.5B weights make a clean small screen possible. | The cached checkpoint is the base model, not the chat-tuned Instruct variant. CPU rollout/training cost is unknown, and simple arithmetic could measure answer formatting more than tool use. | Parked as a fallback. Do not borrow Instruct-model benchmark results for the base checkpoint. |
| Trainer/evaluation reliability | Existing rollback, replay, and audit work already gives concrete engineering evidence; further fault coverage is affordable. | More integrity tests do not by themselves show that an update improves model behavior. | Supporting work only; return to it if no affordable learner study passes feasibility. |

## Feasibility screen and stop rules

The proposed screen will use procedurally generated mini-repositories, not a public benchmark or the historical repair task from the [local-agent pilot](local-agent-pilot.md). It contains 32 episodes but only eight distinct bug templates, each repeated with four generated variants. Treat the aggregate rate as a descriptive feasibility pilot; do not report a binomial confidence interval or imply 32 independent task mechanisms. Report results by family and template. The pilot and its published hidden cases will be permanently excluded from training, development, and confirmation. The four balanced families are: boundary and empty-range semantics; sequence filtering, duplicate handling, and ordering; missing/default/empty aggregation; and order-dependent transformation pipelines. Each family has two bug templates and eight episodes. A later confirmation set must use new repair mechanisms and compositions, not just new names or numbers.

The agent workspace exposes only the task prompt, source files, public cases, and bounded file/edit/test tools. Hidden grader code and cases stay outside the workspace and are not tool-addressable. The edit tool can modify only declared existing files, rejects traversal and symlinks, and applies a unique exact-text replacement. Candidate source is never passed to Python `exec`, a shell, or external test runner: visible and hidden behavior is evaluated by a small AST interpreter that permits only the frozen expression/function subset. Before freeze, an independent stdlib-only auditor regenerates the data, confirms each buggy baseline fails at least one hidden case, confirms a known fix passes all hidden cases, and checks typed output equality. The pilot data will be retained publicly for reproducibility; its hidden cases are private from the agent only during the run and are not a future secret holdout.

Freeze the model revision, task generator, tool schema, prompt, decoding, episode count, resource limits, and pass/fail thresholds before generation. Run offline on CPU with cached weights, at most five model generations and 192 generated tokens per generation, with at most eight total tool calls, under 6 GiB peak RSS and two hours wall time. The candidate proceeds only if:

1. the base passes between 8 and 26 of 32 episodes, inclusive, with success requiring a correct final program, an accepted edit, a visible-test run, and an explicit `finish` call;
2. each of the four task families has at least two passing episodes and each of the eight bug templates has at least one;
3. at least 90% of all attempted calls are schema-valid, authorized tool calls; malformed and unknown calls count in the denominator, zero calls fails, at least 24 episodes must contain both an accepted edit and visible-test run, and at least 24 must finish explicitly;
4. no unsafe or unauthorized tool attempt occurs, all 32 unique task IDs have complete records, and the screen stays within the declared memory and wall-time budget.

If any condition fails, retire this model/task pairing without lowering thresholds or reusing the pilot. A passing screen still does not establish that multi-seed online RL is affordable: estimate the learner cost in a separate bounded smoke before committing to a formal study. Do not switch to a capability claim based on preference NLL, reward alone, or a single favorable seed. Any result from this generated environment must be described as a narrow synthetic code-repair result, not a public benchmark or general coding gain.

## Outcome

The frozen screen completed on 2026-10-09 and failed its task, tool-use and safe-action gates: 0/32 episode successes, zero accepted edits/tests/finishes, 0/69 authorized schema-valid calls, and 60 unsafe or unauthorized attempts. The resource gate passed at 551.7 seconds and 3.20 GiB peak RSS. An independent auditor reconstructed the dataset, trajectory actions, grader outcomes, counters, and decision. The pairing is retired; no learner-cost smoke or training follows. See the [screen report](cpu-code-repair-feasibility-v1-report.md) and [retained bundle](../results/cpu-code-repair-feasibility-v1/run-1/).

## Evidence boundary

The screen itself is not a learning result. Even a successful later confirmation would support only the tested model, task distribution, update method, and resource budget. It would not establish frontier-scale performance, general tool competence, or external reproducibility. External clean-clone reproduction and independent technical review remain open requirements.

The Math TIR feasibility screen has not been run and is not part of the evidence record.
