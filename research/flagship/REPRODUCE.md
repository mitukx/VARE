# Reproduction and resumption

## Fresh Development candidate selection and family review

Selection v3 reuses the frozen v4 metadata screen, excludes the two previously selected identities, and chooses by the recorded SHA-256 rank. It selected VLLM #64830 and did not access PR bodies, patches, source, tests, prompts, grader assets, or Held-Out data. The metadata family review remains separate from source qualification; #64830 is still unqualified and unassigned.

Re-run the selector into a new empty output directory (the frozen result directory is immutable):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_select_mlsys_fresh_development_candidate_v3.py
ruff check \
  scripts/select_mlsys_fresh_development_candidate_v3.py \
  tests/test_select_mlsys_fresh_development_candidate_v3.py
python scripts/select_mlsys_fresh_development_candidate_v3.py \
  --output results/mlsys-fresh-development-candidate-selection-v3-reproduction-001
```

The metadata family review filters mixed-split records before extracting task identity. Its first extraction output is retained and marked superseded for path comparison; correction addendum 1 binds the deduplicated rerun. Reproduce into another new directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_review_mlsys_fresh_development_task_family_overlap_v5.py
ruff check \
  scripts/review_mlsys_fresh_development_task_family_overlap_v5.py \
  tests/test_review_mlsys_fresh_development_task_family_overlap_v5.py
python scripts/review_mlsys_fresh_development_task_family_overlap_v5.py \
  --output results/mlsys-fresh-development-task-family-overlap-v5-reproduction-001
```

The review result is metadata-only and moderate-confidence. Public pretraining exposure is **UNKNOWN**. Do not assign the candidate or run inference until a separate source qualification, exposure, and feasible-attempt protocol is frozen and passed.

## Current checkpoint — vLLM #64575 source-level task qualification

As of 2026-10-31, vLLM PR #64575 is **QUALIFIED_SOURCE_LEVEL** on the five frozen request-model cases only. The pinned baseline failed both type-specific error-message assertions while all three controls passed; corrected wrong-mutant v2 failed the one intended message assertion while four controls passed; upstream-fixed emitted five passing case indicators and exited 0. The first baseline invocation failed during pytest logging setup before collection, and wrong-mutant v1 did not meet its mutation precondition. Both failed attempts are retained and hash-bound; neither is treated as candidate-task failure evidence.

The exact inputs, corrected Git-parent interpretation, environments, arm outcomes, and limitations are recorded in `protocols/mlsys_fresh_vllm_64575_source_qualification_v1.json`, its four append-only addenda, and `results/mlsys-fresh-vllm-64575-source-qualification-v1/`. Verify all 30 result files from the repository root:

```bash
shasum -a 256 -c results/mlsys-fresh-vllm-64575-source-qualification-v1/SHA256SUMS
```

The source qualification runner's own tests and lint can be rechecked without invoking the candidate tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_mlsys_vllm_64575_qualification_tools.py
ruff check \
  scripts/prepare_mlsys_vllm_64575_qualification_overlay.py \
  scripts/run_mlsys_vllm_64575_qualification_arm.py \
  tests/test_mlsys_vllm_64575_qualification_tools.py
```

All baseline, wrong-mutant, and fixed arm invocations for this qualification are consumed. Do not rerun them or change their evidence. The historical task is **UNASSIGNED**; no model inference, training, or Held-Out evaluation occurred, and public pretraining exposure is unknown. Any model attempt needs a separate prospective exposure-ledger event and frozen protocol.

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
set -euo pipefail
mkdir -p work
INPUT_DIR="$(mktemp -d work/context-compaction-inputs.XXXXXX)"
REPLAY_DIR="$(mktemp -d work/context-compaction-replay-rerun.XXXXXX)"
git show origin/main:results/mlsys-fresh-development-agent-diagnostic-v1/transformers-49418/agent-attempt-001.json \
  > "$INPUT_DIR/transformers-agent-attempt-001.json"
git show origin/main:results/mlsys-fresh-development-agent-diagnostic-v1/ray-66694/agent-attempt-001.json \
  > "$INPUT_DIR/ray-agent-attempt-001.json"
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507-4bit-snapshot
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 -m scripts.replay_agent_context_compaction \
  --attempt "$INPUT_DIR/transformers-agent-attempt-001.json" \
  --model-path "$MODEL_PATH" \
  --output "$REPLAY_DIR/transformers-49418.json"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python3 -m scripts.replay_agent_context_compaction \
  --attempt "$INPUT_DIR/ray-agent-attempt-001.json" \
  --model-path "$MODEL_PATH" \
  --output "$REPLAY_DIR/ray-66694.json"
REPLAY_DIR="$REPLAY_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

rerun = Path(os.environ['REPLAY_DIR'])
evidence = Path('results/agent-context-compaction-replay-v1')
for name in ('transformers-49418.json', 'ray-66694.json'):
    actual = json.loads((rerun / name).read_text())
    expected = json.loads((evidence / name).read_text())
    assert actual == expected, name
    assert actual['model_generations'] == actual['grader_calls'] == 0
print('reconstructed prompt records match retained replay evidence')
PY
(cd results/agent-context-compaction-replay-v1 && shasum -a 256 -c SHA256SUMS)
```

The expected result is exact agreement between reconstructed and recorded token counts on every historical turn, followed by `context_fits: true` at 2,765 tokens for Transformers and 2,795 for Ray under the 4,096-token cap. This replay is not a model capability result; the agent may still fail to use the retained context productively.

## Verify the next Development candidate's metadata-only overlap review

The review is limited to frozen candidate metadata and Development/Train exposure records. It does not open task source or tests, run an environment, allocate a split, or perform inference. The candidate remains unqualified and unassigned.

From the repository root, verify the report checksums and all frozen inputs against the exact main commit bound by the protocol:

```bash
set -euo pipefail
python3 -m json.tool protocols/mlsys_fresh_development_task_family_overlap_v2.json >/dev/null
python3 -m json.tool results/mlsys-fresh-development-task-family-overlap-v2/review.json >/dev/null
(cd results/mlsys-fresh-development-task-family-overlap-v2 && shasum -a 256 -c SHA256SUMS)
python3 - <<'PY'
import hashlib
import json
import subprocess

protocol = json.load(open('protocols/mlsys_fresh_development_task_family_overlap_v2.json'))
for item in protocol['frozen_inputs']:
    raw = subprocess.check_output(['git', 'show', f"{protocol['frozen_against_main_sha']}:{item['path']}"])
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], item['path']
sample_path = 'results/mlsys-fresh-development-metadata-review-sample-v1/review-sample.json'
sample = json.loads(subprocess.check_output(['git', 'show', f"{protocol['frozen_against_main_sha']}:{sample_path}"]))
row = next(row for row in sample['candidates'] if row['repository'] == 'huggingface/transformers' and row['number'] == 49458)
canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
assert hashlib.sha256(canonical).hexdigest() == protocol['candidate_identity']['candidate_row_canonical_sha256']
print('nine frozen inputs and the bound candidate row verify; no task content was read')
PY
```

The row-level Development/Train comparisons, confidence, and limitations are recorded in `results/mlsys-fresh-development-task-family-overlap-v2/review.json`. A `NO_PRIOR_FAMILY_OVERLAP_FOUND` result supports source qualification only; it is not task qualification, split allocation, or pretraining-independence evidence.


## Verify and execute Transformers #49458 source qualification

The candidate remains unassigned. This protocol is source/evaluator qualification only and authorizes no model inference. The protocol was committed and pushed before its first candidate test. Check its SHA-256 before preparing or running an arm:

```bash
python3 -m json.tool protocols/mlsys_transformers_49458_source_qualification_v1.json >/dev/null
shasum -a 256 protocols/mlsys_transformers_49458_source_qualification_v1.json
shasum -a 256 work/transformers-49458-inputs/transformers-base-429e6916.tar.gz \
  work/transformers-49458-inputs/video_processing_auto.base-merge-base.py \
  work/transformers-49458-inputs/video_processing_auto.fixed.py \
  work/transformers-49458-inputs/test_video_processing_auto.fixed.py \
  work/transformers-49458-inputs/issue-49451.md
docker --context colima image inspect vare/fresh-qualification-transformers:0.3 --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

Expected protocol SHA-256: `22041c1ef3084b8c47aa091b11aa3b1e4daa1ea50e91248fadb55cd7a176e8b2`. Expected input hashes and the image ID are in the frozen protocol. The current local input filenames are listed above; rename the pinned test overlay to exactly `test_video_processing_auto.py` in a separate read-only mount outside the source tree. Extract the pinned merge-base archive once, then make independent baseline, mutant, and fixed source trees. The wrong mutant changes only the first built-in loop in `video_processor_class_from_name` from `VIDEO_PROCESSOR_MAPPING.items()` to `VIDEO_PROCESSOR_MAPPING_NAMES.items()`; do not change `_load_backend_class`. The fixed tree replaces only `src/transformers/models/auto/video_processing_auto.py` with the hash-pinned merged version.

The first baseline command as originally written failed during pytest collection because the trusted test imports helper modules from the repository `utils/` tree. Preserve that attempt 001 log; do not count it as a semantic baseline result. Protocol addendum `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/protocol-addendum-1.json` freezes the only runner correction: replace the container template environment `PYTHONPATH=/source/src:/source` with `PYTHONPATH=/source/src:/source:/source/utils`. The trusted test remains mounted read-only at `/trusted` and its bytes remain unchanged. Push the addendum before running the corrected baseline. Run one frozen container command at a time, in the required order, using the `container_template` in the protocol. Capture combined output in `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/attempt-001-<arm>.log` and write a JSON receipt next to each log with the exact argv, protocol/source/test/image hashes, start/end UTC, elapsed seconds, exit code, timeout flag, selected/passed/failed/skipped counts, observed assertion, and gate classification. Enforce the 600-second limit with a monotonic timer; check progress at least once per minute and terminate the container if time expires. Every started pytest consumes that arm. Do not retry.

Run one corrected baseline invocation only. The original collection failure is not a baseline result. The corrected baseline must fail at `load_attribute.assert_not_called`; only then run the wrong mutant, which must be rejected by the target assertion on the legacy image-config subcase; only then run upstream-fixed, which must pass both subcases with no skips. Stop on any inconclusive arm. A completed qualification establishes only source-level CPU regression reproducibility. Do not change the split assignment or run inference.


The first corrected baseline attempt is preserved at `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/attempt-002-baseline.log` with its machine-readable receipt `attempt-002-baseline.json`. It met the target baseline gate (47 mapping-resolution calls; one failed test, zero passes/skips). The video-config case failed first, so the legacy-image config case was not reached. The wrong-mutant arm is now authorized; it must be rejected by the same target assertion on the legacy-image case before the fixed arm can run.


The v1 wrong-mutant log and receipt are preserved at `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/attempt-001-wrong-mutant.*`. The target assertion failed after 47 mapping lookups, but pytest did not expose the filename for the upstream method's `unittest.subTest` loop. Since v1 required legacy-case attribution, do not run its fixed arm. A new protocol must report subtest parameters explicitly and bind fresh arm outputs before further candidate tests.


## Case-aware Transformers #49458 source qualification v2

V1's immutable logs remain unchanged. V2 uses the same upstream test file and method, but a frozen stdlib unittest runner that emits each `subTest` parameter/outcome as `VARE_CASE_RUNNER_RESULT=<JSON>`. Verify the V2 protocol and runner hashes, then execute one container per arm using the v2 `container_template`; mount only `case_runner.py` at `/runner`, the trusted test at `/trusted`, and one arm-specific source tree at `/source`, all read-only. Keep `/source/utils` in `PYTHONPATH` as required by addendum 1.

The v2 schedule is baseline → wrong-mutant → upstream-fixed. Each log must contain records for both `video_preprocessor_config.json` and `preprocessor_config.json`. Baseline and wrong-mutant must report `failed` at the target assertion for both; fixed must report `passed` for both. Any collection error, missing case record, skip, timeout, or different assertion is inconclusive and stops the sequence. V1 results do not substitute for v2 arms. Do not assign a split or run inference.


Prepare v2 arm trees with the frozen source archive and overlays:

```bash
mkdir -p work/transformers-49458-v2/baseline work/transformers-49458-v2/trusted work/transformers-49458-v2/runner
mkdir -p work/transformers-49458-v2/wrong-mutant work/transformers-49458-v2/upstream-fixed
tar -xzf work/transformers-49458-inputs/transformers-base-429e6916.tar.gz \
  -C work/transformers-49458-v2/baseline --strip-components=1
cp -a work/transformers-49458-v2/baseline/. work/transformers-49458-v2/wrong-mutant/
cp -a work/transformers-49458-v2/baseline/. work/transformers-49458-v2/upstream-fixed/
cp work/transformers-49458-inputs/test_video_processing_auto.fixed.py \
  work/transformers-49458-v2/trusted/test_video_processing_auto.py
cp results/mlsys-fresh-development-task-qualification-v3/transformers-49458/case_runner.py \
  work/transformers-49458-v2/runner/case_runner.py
cp work/transformers-49458-inputs/video_processing_auto.fixed.py \
  work/transformers-49458-v2/upstream-fixed/src/transformers/models/auto/video_processing_auto.py
python3 - <<'PY'
from pathlib import Path
path=Path('work/transformers-49458-v2/wrong-mutant/src/transformers/models/auto/video_processing_auto.py')
text=path.read_text()
old='for model_type, extractors_dict in VIDEO_PROCESSOR_MAPPING.items():'
new='for model_type, extractors_dict in VIDEO_PROCESSOR_MAPPING_NAMES.items():'
assert text.count(old)==1
path.write_text(text.replace(old,new,1))
PY
docker --context colima image inspect vare/fresh-qualification-transformers:0.3 --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

Verify each generated source-tree digest against `results/mlsys-fresh-development-task-qualification-v2/transformers-49458/input-manifest.json` before execution. Execute the exact per-arm command by substituting its source directory, trusted directory, runner directory, and result log into the v2 `container_template`. Capture stdout and stderr together. Enforce the 600-second ceiling with a monotonic host timer; check at least once per minute and stop the container on expiry. Write one JSON receipt per arm with exact argv, protocol/runner/source/test/image hashes, start/end UTC, elapsed seconds, exit code, timeout flag, named subtest outcomes, and gate classification. Do not retry.


V2 baseline attempt 001 is recorded in `results/mlsys-fresh-development-task-qualification-v3/transformers-49458/attempt-001-baseline.json` (SHA-256 ef5c864eb1bc119a11600bd2fce2b24cc7481bb30f68072cf1073f29b6dc11cc); raw output is `results/mlsys-fresh-development-task-qualification-v3/transformers-49458/attempt-001-baseline.log` (SHA-256 ecee4766d4c918e36abc71175e1be09748eb263978c1b1ed53dcde5f8bc43566). Both named config subTests failed at the target assertion: 47 calls for the video config and 141 for the legacy-image config, with no errors or skips. The v2 wrong-mutant gate is now open.


V2 wrong-mutant attempt 001 is recorded in `results/mlsys-fresh-development-task-qualification-v3/transformers-49458/attempt-001-wrong-mutant.json` (SHA-256 76e78ba63938bfc21013868504960df91a9a01fc7fc2d2093b8078ead3be5d4a); raw output is `results/mlsys-fresh-development-task-qualification-v3/transformers-49458/attempt-001-wrong-mutant.log` (SHA-256 5db740dacb7d9406da01e860fae420cf977125504473bfecaeb7f3d472a53f0c). Both named cases failed the target assertion after 47 mapping loads, with no errors or skips. The upstream-fixed v2 arm is now authorized.


## Verify Transformers #49458 source qualification v2

From the VARE root, verify every result artifact and check the qualification summary:

```bash
cd results/mlsys-fresh-development-task-qualification-v3/transformers-49458
shasum -a 256 -c SHA256SUMS
python3 - <<'PY'
import json
from pathlib import Path
q=json.loads(Path('qualification.json').read_text())
assert q['status']=='QUALIFIED_SOURCE_LEVEL'
assert q['split_assignment']=='NOT_ASSIGNED'
assert q['model_or_agent_exposure'] is False
assert q['arms']['baseline']['outcome'].startswith('both named subTests failed')
assert q['arms']['wrong_mutant']['outcome'].startswith('both named subTests failed')
assert q['arms']['upstream_fixed']['passed_subtests']==2
assert q['arms']['upstream_fixed']['skipped_subtests']==0
print('source qualification and no-inference boundary verified')
PY
```

This verifies source/evaluator reproducibility only. For the commands that prepared and ran each arm, see the v2 protocol and preceding section. Do not infer model capability or assign the candidate based solely on this result.


## One-shot Transformers #49458 Development attempt

Protocol: `protocols/mlsys_transformers_49458_agent_attempt_v1.json` (SHA-256 `4d6561c9d3abe5c85174deca56e77e34f8338d3da3832d4418d82f74e45007ee`). The task contract, prompt, split mapping, source identity, and grader identity are hash-locked under `results/mlsys-fresh-development-agent-diagnostic-v2/tasks/transformers-49458/`. The trusted upstream test bytes are stored separately under `results/mlsys-fresh-development-agent-diagnostic-v2/trusted-test/`; they are not mounted into the agent workspace.

The candidate workspace is a clean sparse checkout of Transformers base `429e69162e2944922e45c2cea24887d32e96fbe7` with only `src/transformers/models/auto/video_processing_auto.py` present. Before model load, revalidate the local model file manifest, check host memory free percentage >= 50, swap used <= 4 GiB, and disk free >= 5 GiB. If any gate fails, do not infer. Then run the exact `controller.command_argv` in the frozen protocol from the VARE root with its offline environment. The agent has no shell, public test, or grader tool. Save the raw result before using the single `independent_grading.command_argv`. That isolated Docker call has network disabled, read-only source/test/runner/root mounts, and no feedback path.

The first Docker preflight checked only mount layout and SHA-256 values; it did not import or execute the test and used zero grader calls. Its receipt is `results/mlsys-fresh-development-agent-diagnostic-v2/transformers-49458/grader-mount-preflight-001.json`. After the attempt, preserve the transcript, patch, sole grader log/receipt, resource telemetry, exposure append, report, and final `SHA256SUMS`. Do not retry under this protocol. One passing Development family would be a feasibility signal only, not RL or generalization evidence.

### Recorded outcome and post-run controller validation

The one authorized attempt is consumed. Its transcript ends with three incomplete, undispatched edit-tool calls (turns 4–6); there is no candidate patch. After listing and searching, the model's third tool turn read lines 360–400 in the legacy image-processor fallback, an off-target area for the lazy registry lookup. The single independent grade ran on the unchanged baseline and failed both named cases at the expected assertion, so it confirms the negative control and does not grade a model patch. See `results/mlsys-fresh-development-agent-diagnostic-v2/transformers-49458/report.md` and `agent-outcome-summary.json`.

The follow-up runner records unclosed tool calls as `truncated_at_token_limit` or `unterminated_tool_call` and includes their turn numbers. The reminder now favors a short exact `replace_text` call for one contiguous change or a minimal changed hunk. This follow-up has no inference path and must not be used to retry #49458. After checkout, validate it with:

```bash
ruff check scripts/run_local_agent.py tests/test_run_local_agent_tool_format.py
work/ci-py311/bin/pytest -q tests/test_run_local_agent_tool_format.py
git diff --check
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

schema = json.loads(Path('research/flagship/state.schema.json').read_text())
state = json.loads(Path('research/flagship/state.json').read_text())
Draft202012Validator(schema).validate(state)
print('mission state schema: valid')
PY
(cd results/mlsys-fresh-development-agent-diagnostic-v2 && shasum -a 256 -c SHA256SUMS)
```

The local controller test result recorded for this checkpoint is 47 passed, and Ruff plus whitespace validation passed. These checks cover the runner instrumentation only. They do not change the NO-GO capability or RL decision.


## Optional per-turn budget after an incomplete tool call

The local controller change is opt-in. Existing attempt protocols retain their configured per-turn cap when `--incomplete-tool-retry-tokens` is omitted. A future frozen protocol may explicitly set a larger value; the protocol-driven runner forwards it, the local controller applies it only after an incomplete tool call enters the existing no-tool retry path, and the evidence collector checks budget equality plus the per-turn expansion chronology. The run record stores the configured value, each effective generation cap, and which turns used an expanded cap. The attempt's `--max-total-new-tokens` remains authoritative.

No model inference is authorized by this controller test record. The consumed Transformers #49458 attempt must not be repeated. Verify the local implementation with:

```bash
ruff check scripts/run_local_agent.py scripts/run_separated_coding_eval.py scripts/record_coding_agent_baseline.py tests/test_run_local_agent_tool_format.py tests/test_record_coding_agent_baseline.py
work/ci-py311/bin/pytest -o addopts='' -q tests/test_run_local_agent_tool_format.py tests/test_mlsys_4b_pilot_protocol.py tests/test_record_coding_agent_baseline.py
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator
schema = json.loads(Path('research/flagship/state.schema.json').read_text())
state = json.loads(Path('research/flagship/state.json').read_text())
Draft202012Validator(schema).validate(state)
print('mission state schema: valid')
PY
(cd results/agent-incomplete-tool-retry-budget-v1 && shasum -a 256 -c SHA256SUMS)
```

This is controller validation, not model-capability evidence. A future model comparison requires its own task, frozen protocol, and authorization.


## Remote integration of incomplete-tool retry budget

PR #156 was merged at 358932bd66e4e78157e5a46307ac182bc68a5c5b. Verify with:

```bash
gh pr view 156 --json state,headRefOid,baseRefOid,mergeCommit,mergedAt
gh pr checks 156
git fetch origin
git rev-parse origin/main
```

The exact-head required checks and calibrated-cpu-tasks passed. The machine-readable CI record is in results/agent-incomplete-tool-retry-budget-v1/controller-change.json; verify its report checksum from that directory with `shasum -a 256 -c SHA256SUMS`.


## RVL-GRPO training-generation task identity

Protocol: `protocols/rvl_grpo_training_generation_task_binding_v1.lock.json`.
The baseline is pinned to `96b3cb86046ccde579caa1c65ccde9f8a2c76e65`; its
append-only run-1 instrumentation erratum is
`protocols/rvl_grpo_training_generation_task_binding_v1_run1_erratum.json`.
Both protocol sidecars and the result bundle are hash-checked below.

Run the candidate regression and the focused adapter suite from the checkout:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short \
  tests/test_rvl_grpo_hooks.py::test_rvl_trainer_boundary_rejects_un_grouped_generation_for_other_task
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short \
  tests/test_rvl_grpo_eval_identity.py \
  tests/test_rvl_grpo_group_boundary.py \
  tests/test_rvl_grpo_group_boundary_calibration.py \
  tests/test_rvl_grpo_hooks.py \
  tests/test_rvl_grpo_inference_cancellation.py \
  tests/test_rvl_grpo_rollout_task_binding.py \
  tests/test_rvl_grpo_shared_module_mode.py \
  tests/test_rvl_grpo_torch_state.py \
  tests/test_rvl_grpo_verifier_trainer_prompt_alignment.py \
  tests/test_rvl_integration.py \
  tests/test_rvl_evaluation_promotion_e2e.py
```

To replay the baseline in a clean worktree with the same regression fixture,
run from the feature checkout while its branch is available:

```bash
git worktree add work/scratch/rvl-grpo-task-identity-base 96b3cb86046ccde579caa1c65ccde9f8a2c76e65
git -C work/scratch/rvl-grpo-task-identity-base checkout codex/rvl-training-generation-task-binding -- tests/test_rvl_grpo_hooks.py
(cd work/scratch/rvl-grpo-task-identity-base && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q --tb=short tests/test_rvl_grpo_hooks.py::test_rvl_trainer_boundary_rejects_un_grouped_generation_for_other_task)
```

The baseline command is expected to fail both cases, each reporting
`train_step_calls=1`. The candidate command is expected to pass both and leave
the call counter at zero. Neither run uses model inference or an external
optimizer. Raw outputs and the result summary are under
`results/rvl-grpo-training-generation-task-binding-v1/`.

```bash
(cd protocols && shasum -a 256 -c rvl_grpo_training_generation_task_binding_v1.lock.json.sha256 rvl_grpo_training_generation_task_binding_v1_run1_erratum.json.sha256)
(cd results/rvl-grpo-training-generation-task-binding-v1 && shasum -a 256 -c SHA256SUMS)
```


## Fresh vLLM #64575 one-shot Development attempt

Frozen protocol: `results/mlsys-fresh-vllm-64575-agent-attempt-v1/protocol.json` (SHA-256 `18f66455cf111708fde170e784cd0a9a07316eacd9db1d5b5b0baf4fa74d8268`). The prospective Development assignment is hash-bound in `results/mlsys-fresh-vllm-64575-agent-attempt-v1/exposure-assignment.json`. The cached model is Qwen2.5-Coder-14B-Instruct-4bit at revision `29efdbab55a161237ab1e432a3abaf6c7ae2b477`; its complete local file manifest is `results/mlsys-fresh-vllm-64575-agent-attempt-v1/runtime/model-file-manifest.json`. The launch preflight has already failed; do not load weights or retry this protocol.

The exact model-attempt argv and independent grader argv are stored in the frozen protocol under `commands`. The attempt command is one-shot, network-denied, and process-tree RSS/time limited. The model workspace contains only the baseline `vllm/entrypoints/anthropic/protocol.py`; the five-case trusted overlay is not exposed to the model. The grader runs once after the agent receipt is durable and never returns feedback to the model.

Pre-protocol resource snapshot: `results/mlsys-fresh-vllm-64575-agent-attempt-v1/runtime/pre-protocol-resource-snapshot.json`. Launch receipt: `results/mlsys-fresh-vllm-64575-agent-attempt-v1/attempt-001/launch-preflight.json` (SHA-256 `d33f329af4268f431e7e21bc7aceb81afe7b179a39af79ce44a98e65dc306a2e`). Frozen inputs and every model snapshot hash matched, but the working-set-plus-headroom check failed: 11,993,432,064 bytes available versus 12,872,896,985 bytes required. Status: **NO-GO / NOT RUN**; the model was not loaded. Do not retry this frozen protocol.


## PR #160 review-fix regression tests

At commit `71aa74ab6a8f6f1625505b68f7f332c49fabff93`, local validation used Python 3.12.12 / pytest 8.4.2:

```bash
ruff check scripts/preflight_local_model_attempt.py scripts/run_bounded_local_agent_attempt.py tests/test_preflight_local_model_attempt.py tests/test_run_bounded_local_agent_attempt.py
/opt/homebrew/Caskroom/miniforge/base/bin/python -m py_compile scripts/preflight_local_model_attempt.py scripts/run_bounded_local_agent_attempt.py tests/test_preflight_local_model_attempt.py tests/test_run_bounded_local_agent_attempt.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -o addopts='' -q tests/test_preflight_local_model_attempt.py tests/test_run_bounded_local_agent_attempt.py
```

Recorded result: 19 passed, Ruff clean, `py_compile` exit 0. Remote exact-head CI was still running when this checkpoint was written. Details: `results/pr160-review-fixes-v1/run-001.json`.

## Fresh vLLM #64830 scoped source qualification

Protocol: `protocols/mlsys_fresh_vllm_64830_source_qualification_v1.json`; ancestry correction: `protocols/mlsys_fresh_vllm_64830_source_qualification_v1_addendum_1.json`. The scope remains the unchanged upstream HiSparse and watermarking cases exercised against extracted baseline, mutant, and fixed `VllmConfig` methods; custom-speculator behavior is excluded.

Overlay preparation on a clean source checkout (the existing evidence directory is immutable and the preparation script refuses to overwrite it):

```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/prepare_mlsys_vllm_64830_qualification_overlay.py \
  --source-repository /private/tmp/vare-vllm-64830-source \
  --protocol protocols/mlsys_fresh_vllm_64830_source_qualification_v1.json \
  --output-dir results/mlsys-fresh-vllm-64830-source-qualification-v1/pretest
```

Before execution, verify the addendum sidecar and reproduce the runner-tooling checks:

```bash
(cd results/mlsys-fresh-vllm-64830-source-qualification-v1 && shasum -a 256 -c protocol-addendum-3-corrected-logging.json.sha256)
ruff check scripts/run_mlsys_vllm_64830_qualification_arm.py tests/test_mlsys_vllm_64830_qualification_tools.py
/opt/homebrew/Caskroom/miniforge/base/bin/python -m py_compile scripts/run_mlsys_vllm_64830_qualification_arm.py tests/test_mlsys_vllm_64830_qualification_tools.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -p no:cacheprovider -o addopts= -q tests/test_mlsys_vllm_64830_qualification_tools.py
```

After the addendum is committed and pushed, run one arm at a time by selecting the corresponding `runner_argv` array from that JSON and invoking it with the VARE repository as the working directory. Never hand-edit an argv. Do not execute the next arm until the previous gate passes.

The original execution addendum 2 is immutable and consumed by baseline attempt-001. Pytest failed in `pytest_configure` because the logging plugin attempted `/dev/null`, which the deny-by-default Seatbelt profile disallowed. The receipt is `results/mlsys-fresh-vllm-64830-source-qualification-v1/baseline-attempt-001/attempt.json`; its classification is `STOPPED_BEFORE_TEST_COLLECTION`, with zero candidate assertions executed. No later arm ran under addendum 2.

The trusted runner now sets `-o log_file=<arm-output-dir>/pytest-internal.log`; this file stays inside the existing arm-specific writable boundary. The sandbox profiles, source/test/mutant bytes, exact test IDs, resource caps, and behavioral acceptance gates are unchanged. The runner unit regression, Ruff, and `py_compile` commands and logs are saved under `results/mlsys-fresh-vllm-64830-source-qualification-v1/retry-tooling-001/`.

Execution addendum 3 (`protocol-addendum-3-corrected-logging.json`) is preserved but superseded before execution: its attempt-002 commands pointed to profiles that allowed writes only under attempt-001. It was never invoked. Execution addendum 4 (`protocol-addendum-4-output-profile-binding.json`) is authoritative. Verify its `.sha256` sidecar, parent hash, runner/test/source/test/Python hashes, all exact stored `runner_argv` and `inner_pytest_command` arrays, and that each profile's sole writable subpath equals its arm's attempt-002 output directory. The three profiles retain default-deny and network-deny rules. **Commit and push addendum 4 and those profiles before any candidate test retry.** Then execute only the stored baseline `runner_argv` once. It must produce exactly the two post-override assertion failures, with successful collection/import and no skips, timeout, resource or isolation breach. If this gate passes, execute the stored wrong-mutant argv once and require the same failures. Only then execute upstream-fixed once and require both cases to pass. Any unexpected result stops the sequence; there is no retry under addendum 4.

Baseline attempt-002 passed the frozen gate under Addendum 4: both cases failed at the post-override assertion on line 73; pytest reported 2 failed, 0 passed/skipped/errors, elapsed 0.53 s, peak RSS 3,932,160 bytes, no timeout. The attempt receipt, dispatch, launcher output, logs, classification, and `SHA256SUMS` are under `results/mlsys-fresh-vllm-64830-source-qualification-v1/baseline-attempt-002/`. Wrong-mutant attempt-002 also passed: both cases were rejected at line 73 with no skips/errors, timeout, or RSS breach. Its receipt and checksums are under `results/mlsys-fresh-vllm-64830-source-qualification-v1/wrong-mutant-attempt-002/`. Upstream-fixed attempt-002 passed both cases (2 passed, 0 failures/skips/errors); no timeout or RSS breach. The baseline, wrong-mutant, and fixed arms all passed their frozen gates. See `qualification.md` and `qualification.json`; verify all 58 bundle files with `shasum -a 256 -c SHA256SUMS`. The result qualifies only the scoped method-level cases; the task remains unassigned and no inference is authorized by these source-level results.

No split assignment, model inference, training, or Held-Out access is authorized. Public pretraining exposure remains UNKNOWN. A source-level pass cannot establish agent capability or RL improvement.


## vLLM #64830 — one-shot Development protocol (frozen, not launched)

Protocol: `results/mlsys-fresh-vllm-64830-agent-attempt-v1/protocol.json` (SHA-256 `9d7a89f8491bdb3096f302ee1349bdcbd0eef6b68abe3967d637cdd76494ae49`). It pins the Qwen2.5-Coder-14B 4-bit MLX snapshot, source/test/worker hashes, one allowed source file, no network, a 4,096-token context, 2,048 generated-token total, 600-second wall limit, and a 12-GiB process-tree RSS limit. The launch preflight and attempt commands are stored as argv arrays in the protocol.

The protocol was locally hash-checked against its sidecar and 24 declared frozen inputs; source qualification and the source-exposure record were independently checked as Git objects at parent revision `a1bc5dc255e38ad1741f58399504fdaf357eae1a`. This static verification did not hash model weights or run launch preflight. The model sandbox isolation probe and Metal smoke passed without loading weights. Independent grader controls are under `results/mlsys-fresh-vllm-64830-agent-attempt-v1/calibration/`.

Run the focused checks:

```bash
shasum -a 256 -c results/mlsys-fresh-vllm-64830-agent-attempt-v1/protocol.sha256
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_mlsys_vllm_64830_agent_grader_v1.py
python -m ruff check scripts/run_mlsys_vllm_64830_agent_grader_v1.py scripts/run_mlsys_vllm_64830_agent_method_worker_v1.py tests/test_mlsys_vllm_64830_agent_grader_v1.py
python -m py_compile scripts/run_mlsys_vllm_64830_agent_grader_v1.py scripts/run_mlsys_vllm_64830_agent_method_worker_v1.py tests/test_mlsys_vllm_64830_agent_grader_v1.py
```

Only run the frozen `preflight_command` after this protocol is remotely pushed and exact-head CI is reviewed. If the receipt is `NO_GO`, preserve it as `NOT RUN`, do not load weights, and do not retry. If it passes, append a prospective assignment event to the append-only vLLM #64830 exposure record and push that event before the one frozen agent command. Push the resulting attempt record and candidate diff before running the one independent grader command. Return no grader output to the model.

No model inference, repair, independent grade, training, or policy update has run under this protocol. No capability, RL, generalization, or algorithmic novelty claim follows from the controls.
