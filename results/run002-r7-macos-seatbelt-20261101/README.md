# Run-002 R7 macOS Seatbelt result

Status: the full R7 build and task-validation sweep completed. The held-out freeze was refused under its fixed 26-repository, 8-task floor. No held-out manifest was produced.

## Result

- 35 candidate repositories were finalized on the pinned order.
- 12 repositories qualified with 8 valid tasks each (96 tasks total).
- 19 ended with `insufficient_valid_tasks`; 4 ended with `env_failed`.
- The runner attempted 903 tasks: 103 passed the reference validation, 769 ended as `reference_tests_missing`, and 31 as `reference_fails`.
- The freeze gate returned exit code 3 and recorded: `12 repos qualify with 8 tasks; floor is 26 x 8`.
- The qualifying pool is 14 repositories (112 tasks) below the required floor. The threshold was not changed.

The 12 admitted repositories are `cool-RR/PySnooper`, `alecthomas/voluptuous`, `mewwts/addict`, `sqlfluff/sqlfluff`, `hukkin/tomli`, `jd/tenacity`, `erikrose/parsimonious`, `weaveworks/grafanalib`, `scrapy/scrapy`, `pudo/dataset`, `tkrajina/gpxpy`, and `pdfminer/pdfminer.six`. `tobymao/sqlglot` had 7 valid tasks and therefore did not qualify.

The gzip-compressed report contains the per-repository outcomes, each successfully built environment's installed distribution inventory, and task identifiers, counts, and hashes. It does not contain problem statements, patches, pytest node names, or raw test output. The four environment-install failures are `conan-io/conan`, `iterative/dvc`, `facebookresearch/fvcore`, and `rsalmei/alive-progress`; their sanitized diagnostics are in the report. Decompress `build_check_report.json.gz` before using it with the freeze script.

A separate post-run diagnostic rechecked one `python-dotenv` candidate under the same Seatbelt runner. Both the bug and reference grades returned pytest exit 4, with all 121 requested IDs unreported; the captured output matched the safe category `missing_module`. This diagnostic is not part of the 903-task totals. Raw stdout, task IDs, and test names were not retained. It indicates an import/collection problem for that example; the main report's reason code does not identify a root cause.

## Reproduction record

The run started at `2026-10-31T22:51:31Z` and finished at `2026-10-31T23:37:01Z`. It used the fixed Run-002 seed and criteria, 8 tasks per repository, no per-repository attempt cap, `--min-free-gib 4`, and `--prune`.

```sh
PYTHONPATH=src python3.12 scripts/run002_build_check.py \
  --cache <persistent-run-cache> \
  --dataset-cache <pinned-v1-dataset-cache> \
  --report <build-check-report.json> \
  --min-free-gib 4 --prune

PYTHONPATH=src python3.12 scripts/run002_freeze_heldout.py \
  --report <build-check-report.json> \
  --out <new-empty-evidence-directory>
```

The host was macOS 27.0.1 (`arm64`, model `MacBookPro18,3`), with 10 CPUs and 32 GiB RAM. The run used CPython 3.12.12, pytest 8.3.5, Apple Git 2.54.0, and the `macos-seatbelt-v1` backend. The full per-repository installed package inventories and environment measurements are in `build_check_report.json`.

The build checker ran package installation in a sanitized Seatbelt process with outbound network access for pip. Grading ran with network denied. The sandbox denied operator-home access, kept the venv read-only during grading, and kept the repository `.git` directory outside the grader workspace, replacing it with a pinned-commit stub. Repository-controlled writes to `.git` were denied, and host Git invocations disabled hooks with `core.hooksPath=/dev/null`. A macOS integration check exercised a malicious `post-checkout` hook and attempted `.git` writes; the hook did not run and the writes were denied. The frozen v1 `RepoEnv` source remained byte-identical (`bc9c8858f39769d6a42d44da0659680ac50b7b5f8cb03573652fac60da2a8d09`).

These results measure environment builds and reference-task validity only. They do not establish model improvement or authorize a downstream learning run. The grader's same-interpreter verdict-forgery residual remains open; see the trusted-grader audit and its follow-up PR.

## Integrity

`refusal.json` records the freeze decision and the SHA-256 of the uncompressed `build_check_report.json` (`b36570ce4e3cc53e3d53eee7623b90b5bf85ccc250d75157e0de87ba9d72c32f`). Decompress the gzip payload and compare its SHA-256 with that value to verify the freeze input. `SHA256SUMS.txt` covers the compressed report, refusal, and this README.
