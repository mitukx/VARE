# Next study decision — 2026-10-09

## Decision

Make a generated multi-turn code/engineering repair task the next primary candidate. First run a frozen, no-update feasibility screen on the locally cached Qwen2.5-0.5B-Instruct model. Use 32 small generated mini-repository tasks across four bug families, with bounded `list`, `read`, `search`, `edit`, and visible-test tools. Keep hidden tests in a separate grader. Measure episode success, tool-call validity, CPU throughput, peak memory, and whether there is enough room to improve. Do not start training unless the task and resource gates pass.

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

The proposed screen will use fresh procedurally generated mini-repositories, not a public benchmark or the historical repair task from the [local-agent pilot](local-agent-pilot.md). The 32-row pilot will be permanently excluded from training, development, and confirmation. Four bug families will cover distinct code changes and tool interactions; the screen will use eight tasks per family. A later holdout must change mutation/composition combinations rather than only replacing names or numbers. The grader should independently rebuild the expected behavior and hidden tests from each episode seed.

Freeze the model revision, task generator, tool schema, prompt, decoding, episode count, resource limits, and pass/fail thresholds before generation. Run offline on CPU with cached weights, at most five tool turns and 192 generated tokens per turn, under 6 GiB peak RSS and two hours wall time. The candidate proceeds only if:

1. the base passes at least 8 of 32 hidden-test episodes while retaining meaningful headroom;
2. each of the four task families has at least one passing episode;
3. at least 80% of tool calls satisfy the frozen schema and file/state constraints;
4. the screen stays within the declared memory and wall-time budget.

If any condition fails, retire this model/task pairing without lowering thresholds or reusing the pilot. A passing screen still does not establish that multi-seed online RL is affordable: estimate the learner cost in a separate bounded smoke before committing to a formal study. Do not switch to a capability claim based on preference NLL, reward alone, or a single favorable seed. Any result from this generated environment must be described as a narrow synthetic code-repair result, not a public benchmark or general coding gain.

## Evidence boundary

The screen itself is not a learning result. Even a successful later confirmation would support only the tested model, task distribution, update method, and resource budget. It would not establish frontier-scale performance, general tool competence, or external reproducibility. External clean-clone reproduction and independent technical review remain open requirements.

The Math TIR feasibility screen has not been run and is not part of the evidence record.
