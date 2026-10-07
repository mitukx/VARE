# Ten-minute technical walkthrough

This walkthrough is for readers who want to inspect the evidence before running a new experiment. It needs Python >=3.9 and Git, with no Python package installation. Grader execution requires POSIX; reading the JSON artifacts needs no network or GPU.

## 1. Verify the retained claims (two minutes)

From the repository root:

```bash
python3 scripts/audit_scheduler.py results/cpu-scheduler-v1
```

Expected: `status: verified`, `historical_evaluations: 80`, `reliability_cases: 16`, and `median_paired_speedup: 3.4857076335411525`. This reconstructs the report from records rather than executing new graders. Timing in a new experiment is not expected to equal these retained numbers.

Read [the frozen protocol](../protocols/cpu_scheduler_v1.json) and [the report](scheduler-report.md). Compare the first pair with the others; the slower initial serial result was retained.

## 2. Follow one accepted result (two minutes)

Open [one fixed HF record](../results/cpu-scheduler-v1/pair-0/workers-4/jobs/task-0-fixed-0/record.json) and its [raw grader output](../results/cpu-scheduler-v1/pair-0/workers-4/jobs/task-0-fixed-0/stdout.bin). Check:

- task ID and fixed source revision;
- source-file and diff hashes;
- descriptor and evaluator hashes;
- `passed`, empty failures and exit 0;
- queue time, execution time and output hashes.

Compare [its pre-fix counterpart](../results/cpu-scheduler-v1/pair-0/workers-4/jobs/task-0-baseline-0/record.json): a valid `candidate_rejected` is different from an operational error. The upstream fixes are calibration material; these records are not an agent discovering the fixes.

## 3. Try falsifying the runner (two minutes)

```bash
python3 -m unittest discover -s tests -p 'test_runner.py' -v
python3 -m unittest discover -s tests -p 'test_durable.py' -v
```

Tests use tiny locally constructed Git repositories. No source fetch, model call or accelerator is involved. Inspect [the source-mutation case](../results/cpu-scheduler-v1/reliability/jobs/mutate/record.json), [output flood](../results/cpu-scheduler-v1/reliability/jobs/flood/record.json), and [inherited-pipe timeout](../results/cpu-scheduler-v1/reliability/jobs/inherited_pipe/record.json). None can authorize candidate success.

## 4. Inspect the implementation (four minutes)

Read `src/vare/runner.py` in this order: `protocol`, `fingerprint`, `execute`, `classify`, `run_campaign`, `audit`. The [execution contract](execution.md) explains each boundary.

Questions that should be answerable from the code and records:

1. Why must a passing JSON object agree with exit status and all observed provenance?
2. What happens when a parent exits but its child keeps stdout open?
3. How are memory used for captured output and concurrent workers bounded?
4. Which events survive a caught interruption? Which machine crashes are not covered?
5. What can a coordinated edit to the manifest, journal and Git trust anchor still fake?
6. Which files are fingerprinted, and why is that insufficient for hostile-code containment?
7. Why is the median of paired ratios different from the ratio of median wall times?
8. What extra evidence would be needed for a claim about model capability or a distributed fleet?

For a new candidate, use the plan and commands in [execution.md](execution.md). For a full protocol reproduction, use the commands in [scheduler-report.md](scheduler-report.md).
