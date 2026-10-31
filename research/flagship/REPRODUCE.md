# Reproduction and resumption

Run commands from the VARE repository root. The evidence includes the prior one-attempt Train-only MLX-LM #1904 trajectory and the two one-attempt fresh Development coding-agent attempts on Transformers #49418 and Ray #66694. All three attempt slots are consumed; do not rerun them. No model update, SFT, GRPO/RL, or Held-out evaluation was authorized or executed.

## Test current implementation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q \
  tests/test_development_repository_repair_campaign_v1.py \
  tests/test_development_repository_repair_campaign_v2.py \
  tests/test_isolated_candidate_expected_exception.py

for py in 311 312 313; do
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
    "work/ci-py${py}/bin/python" -m pytest -q \
    tests/test_mlsys_repo_task.py \
    tests/test_mlsys_deep_qualification_exposure.py
done

ruff check \
  src/vare/mlsys_repo_task.py \
  tests/test_mlsys_repo_task.py \
  scripts/run_mlxlm_1904_calibration.py \
  scripts/red_team_mlxlm_1904.py \
  scripts/audit_mlsys_deep_qualification_exposure_v1.py \
  tests/test_mlsys_deep_qualification_exposure.py

python -m py_compile \
  src/vare/mlsys_repo_task.py \
  scripts/run_mlxlm_1904_calibration.py \
  scripts/red_team_mlxlm_1904.py \
  scripts/audit_mlsys_deep_qualification_exposure_v1.py
git diff --check
```

## Reconcile prior task exposure

The auditor verifies both pinned input hashes, maps task IDs to source PR and family, checks per-stage attempt totals, and refuses to overwrite its output.

```bash
PYTHONPATH=src python scripts/audit_mlsys_deep_qualification_exposure_v1.py \
  --stage3-ledger ../../outputs/VARE-stage3-2026-10-31/task-exposure-ledger.json \
  --output work/exposure-audit-copy.json

shasum -a 256 results/mlsys-task-deep-qualification-v1-exposure-append-v1.json
```

Do not relaunch #45852, #1912, or #1971 as fresh Development tasks. This audit is retrospective and does not reallocate them to Train or Held-out.

## Reproduce the Train-only #1904 grader calibration

Requires a clean VARE commit containing protocol v11, Colima with the pinned image already cached, and a clean `mlx-lm` checkout containing both source revisions. The run script refuses to execute from a dirty VARE worktree, overwrites no evidence directory, and is restricted to the frozen Train task. It authorizes no model rollout.

```bash
git -C ../mlx-lm-source cat-file -e \
  2af5f7ce8620115bf37fbbddf9817f083118df59^{commit}
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904-replay" \
  2af5f7ce8620115bf37fbbddf9817f083118df59

mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904-replay \
  --output results/mlsys-repo-eval-v1/run-17-replay/calibration.json

shasum -a 256 \
  results/mlsys-repo-eval-v1/run-17-replay/calibration.json \
  results/mlsys-repo-eval-v1/run-17-replay/calibration.red-team.json
```

Run 17 passed on implementation commit `36af6f17908b50cb53bb159013eecfa8b0263da0`: baseline failed, known fix passed with exact source identity, source swap was rejected, all nine red-team controls passed, and the symlinked parent was rejected before any grader call. Its manifest SHA-256 is `921e1260451bf94d9a7d8f19e3f62fed4b5b2c3dfec6d09aeca71ab0756c9ed6`. Verify run evidence with `(cd results/mlsys-repo-eval-v1/run-17 && shasum -a 256 -c SHA256SUMS)`, and verify the cumulative root index with `shasum -a 256 -c results/mlsys-repo-eval-v1/SHA256SUMS` from the VARE root. Run 17's affected 23-test suite passed on Python 3.11, 3.12, and 3.13. The v11 protocol and evidence remain immutable.

The PR #151 code review found that opening an unwritten FIFO with blocking `O_RDONLY` could stall host-side source validation before the Docker timeout. Protocol v12 adds nonblocking leaf open and a tenth probe, `workspace_fifo_probe`; it is frozen at SHA-256 `36c5932ab46e00573535f1d6826396cc24cc5021703566f49b9169cad45460c4`. Run 18 **PASSED** on implementation commit `a7d39bda1da662c18bc7b6738b95f605b679ee30`: both task arms matched expectations, the source swap was rejected, and all 10 probes passed. The FIFO was rejected in `0.00012s` without invoking the grader. Manifest SHA-256: `24e4ed3e7e5749a3bd19bdf721cc54104fae913d8902352487a9f3f8801b7d4a`. The focused 25-test suite passed on Python 3.11, 3.12, and 3.13. The run-18 evidence is separate from and does not modify run 17.

```bash
git -C ../mlx-lm-source worktree add --detach \
  "$PWD/../mlx-lm-baseline-1904-v12" \
  2af5f7ce8620115bf37fbbddf9817f083118df59
mkdir -p ../colima-tmp
TMPDIR="$PWD/../colima-tmp" PYTHONPATH=src DOCKER_CONTEXT=colima \
  work/ci-py311/bin/python scripts/run_mlxlm_1904_calibration.py \
  --source-root ../mlx-lm-baseline-1904-v12 \
  --output results/mlsys-repo-eval-v1/run-18/calibration.json
shasum -a 256 \
  results/mlsys-repo-eval-v1/run-18/calibration.json \
  results/mlsys-repo-eval-v1/run-18/calibration.red-team.json
```

The 25-test focused suite passed locally on Python 3.11, 3.12, and 3.13. The v12 acceptance gate preserves all run-17 conditions and requires the FIFO to be rejected in under one second without invoking the candidate grader.

## Full CI and PR state

PR #151 is integrated at `818fd8c802277ad360ee9b5250d0315cfb8dad81`; PR #150 is in its history at `cb5c9d91ce269014b4166644248aa2b71a5c3dae`. PR #151 exact-head Python 3.11/3.12/3.13, lint, typecheck, CPU evidence, and E2E recovery checks passed. PR #152 contains the current audit and run-19 evidence; its exact-head checks are pending. See `state.json` for the verified merge and current PR.

## Run 19 — one Train-only model trajectory

Protocol `protocols/mlsys_task_model_attempt_mlxlm_1904_v1.json` was committed before inference. It pins one greedy Qwen3-4B-Instruct-2507-4bit MLX attempt on the existing Train task `mlxlm-1904-tool-arguments`, base `2af5f7ce8620115bf37fbbddf9817f083118df59`, one allowed file, 2,048 total generated tokens, 12 turns, 300 seconds, and one independent v12 grader call. The exact model snapshot, candidate prompt/spec, controller, and grader hashes are recorded in the protocol and run manifest.

The run completed in 31.679 seconds using 302 generated tokens over five tool turns. The independent grader passed `dict_arguments` and `json_string_control` (2/2); no grader feedback was returned to the agent. Evidence is retained in `results/mlsys-repo-eval-v1/run-19/`. A first infrastructure launch loaded weights but failed before prompt rendering or token generation because a linked worktree had a `.git` pointer file. That launch error is preserved separately and did not consume the single inference attempt. The successful attempt is consumed; do not rerun it or use grader feedback for a second attempt.

Verify the immutable run bundle and cumulative checksum index:

```bash
(cd results/mlsys-repo-eval-v1/run-19 && shasum -a 256 -c SHA256SUMS)
shasum -a 256 -c results/mlsys-repo-eval-v1/SHA256SUMS
python - <<'PY'
import json
from pathlib import Path
attempt = json.loads(Path('results/mlsys-repo-eval-v1/run-19/agent-attempt-1.json').read_text())
grade = json.loads(Path('results/mlsys-repo-eval-v1/run-19/independent-grade.json').read_text())
assert attempt['new_tokens'] == 302 and attempt['stop_reason'] == 'agent_finished'
assert grade['grader_call_count'] == 1 and grade['grader']['passed'] is True
assert [case['passed'] for case in grade['grader']['cases']] == [True, True]
PY
```

This is one independently graded task from Train. It is not Development/generalization evidence, a policy update, SFT, GRPO, or RL. The public task's pretraining exposure is unknown. The grader covers the selected function contract, not the full upstream HTTP/model-serving test. The appended exposure event is in `run-19/exposure-append.json`; the prior Stage 2/3 ledgers and frozen queue audit remain unchanged.

## Model, receipt, and resume status

- The prior Qwen3-4B authored-task cohort has one verified success in two tasks; it produced no policy-update checkpoint.
- Run 19 adds one narrow Train-task pass (2/2 cases). Its transcript, edit, independent receipt, failed pre-inference launch, model snapshot hashes, and resource telemetry are retained.
- The queue audit finds zero fresh Development candidates; the historical Held-out allocation remains untouched.
- The Stage 3 candidate receipt remains at `../../outputs/VARE-stage3-2026-10-31/capability-diagnostic-exact-replace-v4/grader-receipts/stage3_qwen3_4b_exact_replace_v4__mlxlm-1912-top-p-min-token.terminal-lf-regrade.json`; it is `wrong_output` on 0/14 correctness cases.
- No resumable model or optimizer run exists. Do not relaunch the consumed task/model pairs.

Next, integrate the audit and run-19 evidence by focused PR. Then prospectively source-qualify a diverse Development cohort without opening Held-out files or tests.

## Fresh Development task qualification — Transformers #49418

The selected task and six parameter cases are frozen in `results/mlsys-fresh-development-task-qualification-v1/protocol-addendum-4.json` (SHA-256 `6be83fbf944ee60467ed87d30e5eebeb2f8ce4af8096235908a160e17d8b2598`). Attempt 3 is an inconclusive runtime failure, not a baseline result. Runtime v3 is pinned by image ID `sha256:d93a01090a15c8c9ce47d7e08c6bc935f1abb83683f6318dff866a07f08f8ff2`; its Dockerfile and build log are under `results/mlsys-fresh-development-task-qualification-v1/environment/`.

Prepare two separate `/source` trees from the frozen base (`bd3f926f52b55a0a59e812376e7624ca9d471ac2`) and fixed (`1734d8d9d6a4c5331ba87dd4a1a36d3a3cb25ac1`) commits. Mount the fixed commit's `tests/utils/test_cache_utils.py` at `/trusted/test_cache_utils.py` in both runs. Verify the frozen source/test input manifest before launch. Use the six exact node IDs in addendum 4, the resource/security flags in addenda 1 and 4, `MAX_JOBS=1`, and read-only source/test mounts. The only additional writable mount is:

```text
/usr/local/lib/python3.12/site-packages/optimum/quanto/library/extensions/cpp/build:rw,exec,nosuid,nodev,size=256m,mode=1777
```

Run the baseline first. Do not run the fixed arm unless the intended residual-boundary correctness assertion executes and fails on baseline. Retain each raw log and a separate JSON attempt record. No candidate prompt, agent/model inference, optimizer update, Development assignment, or Held-out access is included in this qualification protocol.

The baseline attempt under addendum 4 is a valid negative control: 4/6 expected assertions failed and 2/6 passed. A first fixed-arm invocation was rejected before case execution due to a shell selector typo; its log/record are retained as inconclusive. Addendum 5 freezes the exact selector strings and authorizes one fixed-source retry. Build the pytest argv from that `exact_test_node_ids` list rather than retyping it. The retry uses the same runtime v3 image and isolation flags listed above.

A second launcher error is also retained: Docker started but pytest did not because the wrapper passed `python3` twice. This is not a test result. With `--entrypoint=python3`, pass only `-m pytest -q -p no:cacheprovider` plus the six IDs from addendum 5.


### Transformers #49418 paired result

The exact input manifest matched before both arms. Baseline attempt 004: 4 expected failures and 2 passes; fixed attempt 003: 6 passed, zero skipped, pytest 25.27s. The `qualification.json` record (SHA-256 `75bfa210ef7ce8184ade0d5c98c92047d6eab918e6e662d2f12a3ec07d28ea4e`) is the concise paired summary; raw logs and attempt records remain beside it. The two zero-case fixed launcher errors are retained and excluded from the paired outcome. This source-level result does not assign a split or authorize model inference.


### TRL #7668 paired source result

Baseline attempt 001: two target `TypeError` failures from the base function rejecting `gradient_checkpointing_kwargs`; assertions did not execute. Fixed attempt 001: 2 passed, including exactly the permitted tiny-fixture forward/loss/autograd checks. The frozen gate requires a baseline assertion failure, so the pair is retained but the task remains unqualified. Evidence summary SHA-256: `f2187cb5d31a6aa5b1bfceb174e220eae5589de91e1ad1e1ce0b1ce95579a47c`.


### MLX #4778 paired source result

Baseline attempt 001: the exact CPU complex-power test failed the frozen VJP assertion. Fixed attempt 001: 1 passed, zero skipped. The first fixed-image launch used a stale tag and never started a container; the failed launch is retained separately and excluded. Paired qualification SHA-256: `29eeb28838404a8052ab129b52b547651d65c5bd3e8ea45059e4728718fee7ee`.


## Ray #66694 source-level qualification

The ordered baseline, wrong-mutant, and fixed attempts are consumed; do not rerun them without a new frozen protocol. The frozen selectors, exact Docker argv, logs, image identity, and hashes are recorded in `results/mlsys-fresh-development-task-qualification-v1/ray-66694/commands.json`, `protocol-addendum-5-runtime-freeze.json`, and the adjacent attempt/qualification records. The fixed result is 4/4 passed, zero skips. To audit without rerunning:

```bash
(cd results/mlsys-fresh-development-task-qualification-v1/ray-66694 && shasum -a 256 -c SHA256SUMS)
python - <<'PY'
import json
from pathlib import Path
root = Path('results/mlsys-fresh-development-task-qualification-v1/ray-66694')
record = json.loads((root / 'attempt-001-fixed.json').read_text())
qualification = json.loads((root / 'qualification.json').read_text())
assert record['status'] == 'FIXED_PASSED'
assert record['passed_cases'] == record['expected_cases'] == 4
assert record['failed_cases'] == record['skipped_cases'] == 0
assert qualification['status'] == 'QUALIFIED_SOURCE_LEVEL'
assert qualification['split_assignment'] == 'NOT_ASSIGNED'
PY
```

This result is a CPU source-level regression qualification only. It does not establish target-model capability, Development generalization, pretraining independence, SFT/RL gains, or algorithmic novelty.

## Fresh candidate pipeline and mission-state validation

Install the repository test/lint tools, then reproduce the focused local checks:

```bash
python3 -m pip install -e . pytest==8.4.2 ruff==0.15.12 jsonschema==4.25.1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q -rA \
  tests/test_flagship.py \
  tests/test_mlsys_fresh_development_candidate_discovery_v1.py \
  tests/test_mlsys_fresh_development_candidate_discovery_v2.py \
  tests/test_mlsys_fresh_development_candidate_discovery_v3.py \
  tests/test_mlsys_fresh_development_candidate_metadata_sample_v1.py \
  tests/test_mlsys_fresh_development_candidate_screen_v4.py
ruff check \
  scripts/discover_mlsys_fresh_development_candidates_v1.py \
  scripts/discover_mlsys_fresh_development_candidates_v2.py \
  scripts/discover_mlsys_fresh_development_candidates_v3.py \
  scripts/sample_mlsys_fresh_development_candidate_metadata_v1.py \
  scripts/screen_mlsys_fresh_development_candidate_pool_v4.py
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

schema = json.loads(Path('research/flagship/state.schema.json').read_text())
state = json.loads(Path('research/flagship/state.json').read_text())
Draft202012Validator(schema).validate(state)
print('mission state schema: valid')
PY
```

The focused suite passed 32 tests on Python 3.11. The Ray qualification has a separate per-task checksum index; verify it from the repository root with the command in the preceding section. The CI workflow remains the authoritative full-suite and Python 3.11/3.12/3.13 check.

## Development/Train task-family overlap review

The review is metadata-only. It reads the hash-frozen candidate metadata and Development/Train exposure records; it does not inspect Held-Out task material or change any split. Verify the report and every parent/addendum input:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('.')
state = json.loads((root / 'research/flagship/state.json').read_text())
report_path = 'results/mlsys-fresh-development-task-qualification-v1/task-family-overlap-review-v1.json'
ref = next(x for x in state['result_hash_references'] if x['path'] == report_path)
assert hashlib.sha256((root / report_path).read_bytes()).hexdigest() == ref['sha256']
report = json.loads((root / report_path).read_text())
assert report['input_hash_verification'] == 'PASSED_ALL_FROZEN_INPUTS'
assert report['historical_inventory_count'] == 6
assert report['decision']['split_assignments_made'] is False
assert report['decision']['model_inference_authorized'] is False
assert [x['classification'] for x in report['candidate_assessments']] == [
    'NO_PRIOR_FAMILY_OVERLAP_FOUND',
    'POSSIBLE_FAMILY_OVERLAP',
    'NO_PRIOR_FAMILY_OVERLAP_FOUND',
]
print('overlap record hash and frozen review outcome: valid')
PY
```

Transformers #49418 and Ray #66694 may proceed to a separately frozen prospective Development protocol. Keep MLX #4778 unassigned pending replacement or explicit risk handling. These are metadata-level judgments, not pretraining-independence findings.

## Prospective Development model diagnostic v1

Protocol: protocols/mlsys_fresh_development_agent_capability_v1.json (SHA-256 e85f4215ac1c2e64916eba373886c9f0cd24a8a7915442dfe24a173b11fc0c0d). This freezes exactly one offline greedy Qwen3-4B attempt for Transformers #49418 and one for Ray #66694, followed by one independent grade per candidate. The only task source files exposed to the local agent are the allowlisted production paths. Do not run public tests, repeat attempts, or return grader results to the model.

The local venv is isolated under work/; the hash-pinned dependency lock is tracked at results/mlsys-fresh-development-agent-diagnostic-v1/runtime/requirements.lock. The install must use hardlinks: uv's symlink mode made MLX's extension resolve its adjacent library from a cache directory. The existing model snapshot must match runtime/model-file-manifest.json.

Before each attempt, inspect host resources. Abort without loading the model if memory pressure free falls below 50%, swap used exceeds 4 GiB, or free disk falls below 5 GiB.

    memory_pressure -Q
    sysctl vm.swapusage
    df -h .

Run from the VARE repository root. The attempt runner checks its wall budget between generations; stop the process if the 360-second outer limit is reached. Do not rerun after timeout or partial output.

    PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1       work/mlx-qwen3-agent-runtime/venv/bin/python scripts/run_local_agent.py       --workspace "$PWD/work/agent-workspaces/transformers-49418-real"       --model-path /Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b       --model-repository mlx-community/Qwen3-4B-Instruct-2507-4bit       --backend mlx       --task-root results/mlsys-fresh-development-agent-diagnostic-v1/tasks/transformers-49418       --output results/mlsys-fresh-development-agent-diagnostic-v1/transformers-49418/agent-attempt-001.json       --instruction "Inspect the task statement and make the minimal source change it requests. Use the tools to inspect and edit the workspace, then finish."       --seed 49418 --max-turns 12 --max-no-tool-retries 2       --max-new-tokens-per-turn 256 --max-total-new-tokens 2048       --max-context-tokens 4096 --wall-seconds 300       --temperature 0 --top-p 1 --top-k 0 --no-do-sample       --edit-format unified_diff

    PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1       work/mlx-qwen3-agent-runtime/venv/bin/python scripts/run_local_agent.py       --workspace "$PWD/work/agent-workspaces/ray-66694-real"       --model-path /Users/user/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b       --model-repository mlx-community/Qwen3-4B-Instruct-2507-4bit       --backend mlx       --task-root results/mlsys-fresh-development-agent-diagnostic-v1/tasks/ray-66694       --output results/mlsys-fresh-development-agent-diagnostic-v1/ray-66694/agent-attempt-001.json       --instruction "Inspect the task statement and make the minimal source change it requests. Use the tools to inspect and edit the workspace, then finish."       --seed 66694 --max-turns 12 --max-no-tool-retries 2       --max-new-tokens-per-turn 256 --max-total-new-tokens 2048       --max-context-tokens 4096 --wall-seconds 300       --temperature 0 --top-p 1 --top-k 0 --no-do-sample       --edit-format unified_diff

Independent grader argv templates, test hashes, image digests, resource limits, network restrictions, and expected cases are frozen in the protocol. Substitute the repository root for {VARE_ROOT} and redirect each one-shot Docker command's stdout/stderr to its task log. Save one JSON receipt per grader call. Grade only after the corresponding agent record is durable; never feed results back to either attempt.

The first Transformers launch used the original Git worktree and failed before prompt encoding because the controller requires a real .git directory. The exact traceback and zero-token classification are in results/mlsys-fresh-development-agent-diagnostic-v1/transformers-49418/launch-preflight-001.json. Addendum 1 records that no inference slot was consumed and binds replacement workspaces with real .git directories, exact base revisions, allowlisted files, and clean pinned status. Use the replacement workspace paths shown in the commands above. No launch that reaches generate_turn may be repeated.

At protocol freeze, import-only MLX/Metal preflight passed and no model generation had run. The replacement workspace preflights then passed. Both authorized generations and both independent grades have now run once; results are in `results/mlsys-fresh-development-agent-diagnostic-v1/`. Transformers produced 126 tokens and Ray 399 tokens; neither produced a source edit, and both stopped on `context_budget_exhausted`. The independent results were Transformers 4 failed / 2 passed and Ray 2 failed / 2 passed, with zero skips. Never rerun either task.

## Verify the fresh Development diagnostic evidence

From the repository root, verify the evidence checksums and summary records:

```bash
cd results/mlsys-fresh-development-agent-diagnostic-v1
shasum -a 256 -c SHA256SUMS
python3 - <<'PY'
import json
from pathlib import Path

root = Path('.')
for task, expected in [('transformers-49418', (4, 2)), ('ray-66694', (2, 2))]:
    attempt = json.loads((root / task / 'agent-attempt-001.json').read_text())
    grade = json.loads((root / task / 'independent-grade.json').read_text())
    assert attempt['edit_applied'] is False
    assert attempt['candidate_diff'] == ''
    assert attempt['stop_reason'] == 'context_budget_exhausted'
    assert grade['grader_call_count'] == 1 and grade['feedback_to_agent'] is False
    assert (grade['failed_cases'], grade['passed_cases']) == expected
    assert grade['skipped_cases'] == 0
print('both one-shot Development outcomes and single grader receipts: valid')
PY
```

`report.md` summarizes the diagnostic and limitations. `exposure-append.json` records the two Development exposures; `resource-telemetry.json` contains pre-run gates and process metrics. Their hashes are included in `SHA256SUMS` and `research/flagship/state.json`.

## Replay the context-compaction controller on saved attempts

This is a tokenizer-only reconstruction of the already-consumed Transformers #49418 and Ray #66694 transcripts. It checks every recorded per-turn prompt length before applying compaction. It does not load model weights, generate tokens, grade candidates, or modify the original run records. Never rerun either task.

From the repository root, with the pinned local tokenizer snapshot available:

```bash
mkdir -p work/results-context-compaction-inputs results/agent-context-compaction-replay-v1
git show origin/main:results/mlsys-fresh-development-agent-diagnostic-v1/transformers-49418/agent-attempt-001.json \
  > work/results-context-compaction-inputs/transformers-agent-attempt-001.json
git show origin/main:results/mlsys-fresh-development-agent-diagnostic-v1/ray-66694/agent-attempt-001.json \
  > work/results-context-compaction-inputs/ray-agent-attempt-001.json
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507-4bit-snapshot
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 -m scripts.replay_agent_context_compaction \
  --attempt work/results-context-compaction-inputs/transformers-agent-attempt-001.json \
  --model-path "$MODEL_PATH" \
  --output results/agent-context-compaction-replay-v1/transformers-49418.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 -m scripts.replay_agent_context_compaction \
  --attempt work/results-context-compaction-inputs/ray-agent-attempt-001.json \
  --model-path "$MODEL_PATH" \
  --output results/agent-context-compaction-replay-v1/ray-66694.json
(cd results/agent-context-compaction-replay-v1 && shasum -a 256 -c SHA256SUMS)
```

The expected result is exact agreement between reconstructed and recorded token counts on every historical turn, followed by `context_fits: true` at 2,765 tokens for Transformers and 2,795 for Ray under the 4,096-token cap. This replay is not a model capability result; the agent may still fail to use the retained context productively.
