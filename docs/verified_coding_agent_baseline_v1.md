# Verified coding-agent evidence binding v1

**Status: implementation + deterministic controls, no fresh model/agent result.** This is a bounded addition to VARE's existing historical-task and local-agent machinery. It does not modify previously frozen protocols, task packs or historical outcomes.

## Reuse and testable contract

Reuse `scripts/prepare_task.py`, `scripts/run_local_agent.py`, `scripts/grade_task.py`, the `rvl_behavior_policy_parity` E1 grader and its task/brief/lock without copying trusted evaluators into candidate workspaces. The added `scripts/record_coding_agent_baseline.py` is an **offline** collector and auditor: it reads a frozen protocol, the agent JSON record and the independently generated grader JSON, verifies all available task/source/model identities, task protocol hash locks, exact source/diff hash correspondence, allowed diff paths, budgets, step/tool transcript integrity and grader provenance. It then writes an immutable-per-path JSON summary, copies input evidence, and writes a SHA-256 file manifest. Invalid evidence is retained with a failure classification and never counted as a passing task.

The collector does **not** execute candidate Python code, run `grade_task.py`, or provision a sandbox. The existing `grade_task.py` runs candidate code and must be invoked only inside a separately configured, appropriately isolated VM/container with limited filesystem/network and no credentials or hidden evaluator assets accessible to the candidate. A CLI flag alone is not an isolation guarantee. A passing JSON document is not independently signed; this checks consistency and assumes the trusted grader was actually executed in an authorized environment. External attestations remain a separate requirement for hostile contributors.

The supplied E1 historical task has already been used for calibration and model trials. Reusing it for runner integration **cannot** be counted as fresh held-out generalization. New tasks must be separately located, checked for license and pinned failing/fixed revisions, calibrated in an isolated environment, and locked before a model run. This change adds no unverified historical bugs.

## Freeze gate

`protocols/verified_coding_agent_baseline_v1.json` is a **premeasurement** versioned protocol with the checked historical v4 lock SHA-256 (`4996eed5c81863904f4596a6a1e914120bc36c8e5c4cdc0981c3811ec67df8f2`). Before an experiment, independently verify every declared SHA against the clean checkout; commit the protocol and do not alter it after seeing results. If the target historical task's lock has advanced since this snapshot, use a new version, not a silent edit. Run `python3 -m pytest -q tests/test_record_coding_agent_baseline.py` before measuring. This protocol uses independent seeds 101–103 with a maximum of 30 turns and 7,680 generated tokens; other diagnostic horizons (10 or 60 turns) require separately frozen protocol IDs. The task is development/integration only and receives no model-learning capability label.

## Reproduction (only after a suitable external sandbox and local model are available)

From a clean clone of VARE at the recorded revision:

```sh
python3 scripts/prepare_task.py --task-root benchmarks/historical/rvl_behavior_policy_parity --workspace /isolated/agent-101
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/run_local_agent.py \
  --workspace /isolated/agent-101 --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --model-path /models/7ae557604adf67be50417f59c2c2f167def9a775 \
  --instruction 'Read src/rvl_systems/hf_backend.py now. Implement the task with a minimal patch and check syntax.' \
  --output /isolated/out/agent-101.json --seed 101 --max-turns 30 \
  --max-new-tokens-per-turn 256 --max-total-new-tokens 7680 --wall-seconds 900
# Execute ONLY within the separate evaluator isolation boundary:
python3 scripts/grade_task.py --workspace /isolated/agent-101 \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --result /isolated/out/grade-101.json
# Outside the candidate, bind exact raw records; do not overwrite an existing result:
python3 scripts/record_coding_agent_baseline.py \
  --protocol protocols/verified_coding_agent_baseline_v1.json \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --agent-record /isolated/out/agent-101.json --grader-record /isolated/out/grade-101.json \
  --output-dir results/verified-coding-agent-baseline-v1/seed-101
```

Repeat with fresh task workspaces and seeds 102–103. A timeout or grader non-emission must be represented by a failed evidence record, not quietly excluded. The collector returns nonzero for invalid evidence but zero for an otherwise valid negative task result so the audit can preserve it as data.

## What is not yet demonstrated

No new historical issue, known fix, model tool call, candidate patch, trusted grader execution, or learned policy update is claimed by this implementation. The reference runner checks wall time **between** generations, not preemption during a generation. The collector checks byte hashes, not authenticity of a forged pair of records with consistent rewritten hashes. GPU/resource costs are not inferred when absent. The next empirical gate is one isolated end-to-end real agent run on a correctly pinned integration control, followed by independently calibrated **fresh** real-repository tasks before any held-out coding-RL or post-training comparison.

## Execution environment boundary observed during preparation

At preparation time, the connected GitHub read API reported VARE `main` at `23cba21423e961a1f49547b16b6c7858d9bf0ba6` (PR #76 merged). The local execution container could not resolve `github.com` for `git clone` (`Could not resolve host: github.com`), and no isolated VM with a local model snapshot was available through this task environment. Therefore **no** clean-checkout historical grader calibration, protected execution of agent-generated source, full VARE test suite, or real-model attempt was executed here. The 21 passing local tests exclusively exercise the offline artifact-binder with deterministic mock evidence. Do not add them to an E1 task-success numerator. New task sourcing and known-fix calibration remain blocked pending network access and a trusted execution boundary.
