# Reproduction and resumption

Run commands from the VARE repository root. Model inference, training, and grader-calibration experiments are intentionally not launched by these commands.

## Test the current implementation

Run the three files affected by the PR #150 CI regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_development_repository_repair_campaign_v1.py \
  tests/test_development_repository_repair_campaign_v2.py \
  tests/test_isolated_candidate_expected_exception.py -q
```

The full CI test job uses:

```bash
python -m pytest \
  --ignore=tests/test_math500_grader_v1.py \
  --ignore=tests/test_math500_grader_v2.py
```

The two ignored Math500 files run in their own workflow jobs. For a local run that needs Docker, select a running Docker context, for example `DOCKER_CONTEXT=colima` on the machine used for the Stage 3 calibration. GitHub Actions provides its own Linux Docker daemon.

## Reproduce and inspect the current CI diagnostic

The pre-fix PR head `86abbd4df86dcebc539143de3f2f84b1207fff16` failed the same three assertions on all three Python versions. The logged runs were `41105309069` and `41105317116`.

```bash
gh run view 41105317116 --repo mitukx/VARE --log-failed
gh run view 41105309069 --repo mitukx/VARE --log-failed
```

The new PR head must be checked again; these historical run IDs are not evidence that the correction passed CI.

## Inspect and verify retained Stage 3 evidence

The immutable Stage 3 report, exposure ledger, and evidence index are stored outside this repository at `../../outputs/VARE-stage3-2026-10-31/` in the local project workspace. The report records 8 model attempts, 0 independently verified repairs, no fresh families used, no GRPO update, and the untouched held-out PR #1979. Do not rerun those consumed task/model pairs or modify the retained evidence.

```bash
cd ../../outputs/VARE-stage3-2026-10-31
tail -n +2 evidence-sha256.txt | shasum -a 256 -c -
```

The indexed Stage 3 run identifiers, report digest, exposure-ledger digest, and protocol locks are recorded in `research/flagship/state.json`.

## Resume, checkpoint, and receipt status

- **Resume:** No resumable coding-agent or optimizer run exists for the Stage 3 cohort. Its attempts are complete, tasks are already exposed, and they must not be relaunched as fresh work.
- **Model checkpoint:** No task-success coding-policy checkpoint or GRPO checkpoint was produced. Checkpoint verification is therefore NOT APPLICABLE for this cohort; do not promote an unrelated adapter as coding capability evidence.
- **Final grading receipt:** Inspect `../../outputs/VARE-stage3-2026-10-31/capability-diagnostic-exact-replace-v4/grader-receipts/stage3_qwen3_4b_exact_replace_v4__mlxlm-1912-top-p-min-token.terminal-lf-regrade.json`. The independent grader transport succeeded and all 14 correctness cases were `wrong_output`; this is a retained failure, not a passing receipt.

The immediate resumable task is the PR #150 integration step in `STATUS.md`: commit/push the regression correction, inspect the new CI matrix and reviews, merge only when permitted, and verify the resulting SHA on remote `main`.
