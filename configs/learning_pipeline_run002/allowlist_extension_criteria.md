# Run-002 held-out pool: repository admission criteria (frozen)

Status: FROZEN before any task content of the candidate repositories was processed.
The sha256 of this file is stored in `allowlist_extension_criteria.md.sha256`. This file is
never edited after freezing; corrections go into `allowlist_extension_criteria.addendum_N.md`.

Scope: decides which upstream repositories (and which of their tasks) may enter the
run-002 held-out pool. It does not select R or k (that is chosen from verified pass counts
and the power table, before any model is run) and it does not change any pipeline filter.

## 0. Fixed inputs

| input | value |
|---|---|
| pipeline code / config | VARE `origin/main` 6cafa37a: `src/vare/learning_pipeline/{curriculum,sources,exclusions,edit_format,repo_env}.py`, `configs/learning_pipeline_v1/curriculum.json` |
| dataset | SWE-smith-py, revision `77cab9055d42ab4a5c25c89a8f937096db13558e` (11 shards, cached) |
| Train manifest | `configs/learning_pipeline_v1/train.manifest.jsonl` (80 IDs) |
| run-001 held-out manifest | `configs/learning_pipeline_v1/heldout.manifest.jsonl` (42 IDs) |
| exposure ledger | `exposure_ledger_ids.txt` (263 IDs: every SWE-smith instance ID found on any remote branch or in box work files; superset of the 122 manifest IDs; includes 1 `jd/tenacity` ID), sha256 `3ad576efcbf38d43673dd42854f887ed66794a44416f63d2c55d26cc5494f480` |
| licence survey | `meta/smith_repo_licenses_final.json` (2026-10-31; per repo: source `github_api` = code-host licence API SPDX id, or `rebench_v2_field` = per-instance licence field of the pinned SWE-rebench-V2 metadata) |
| pipeline filter values | unchanged: max_changed_lines 30, max_fail_to_pass 30, max_pass_to_pass 400, max_problem_chars 3000, **max_file_chars 8000** |

## 1. Candidate set (closed list)

A. The 24 allowlisted repos (`sources.REPO_LICENSES`) that are not run-001 Train/held-out repos:
alecthomas/voluptuous, arrow-py/arrow, chardet/chardet, cknd/stackprinter, cool-RR/PySnooper,
dbader/schedule, erikrose/parsimonious, google/textfsm, gruns/icecream, hukkin/tomli, jd/tenacity,
keleshev/schema, lepture/mistune, life4/textdistance, madzak/python-json-logger,
martinblech/xmltodict, mewwts/addict, r1chardj0n3s/parse, rustedpy/result, spulec/freezegun,
termcolor/termcolor, theskumar/python-dotenv, un33k/python-slugify, vi3k6i5/flashtext.

B. The 45 non-allowlisted SWE-smith-py repos whose survey licence is in
`sources.PERMISSIVE_LICENSES` and that have >= 8 metadata-eligible disjoint tasks
(licence, survey source):

| repo | licence | source |
|---|---|---|
| Project-MONAI/MONAI | Apache-2.0 | github_api |
| alanjds/drf-nested-routers | Apache-2.0 | github_api |
| amueller/word_cloud | MIT | github_api |
| bottlepy/bottle | MIT | github_api |
| burnash/gspread | MIT | github_api |
| cantools/cantools | MIT | github_api |
| conan-io/conan | MIT | rebench_v2_field |
| cookiecutter/cookiecutter | BSD-3-Clause | github_api |
| dask/dask | BSD-3-Clause | rebench_v2_field |
| datamade/usaddress | MIT | github_api |
| django/channels | BSD-3-Clause | rebench_v2_field |
| django/daphne | BSD-3-Clause | github_api |
| facebookresearch/fvcore | Apache-2.0 | github_api |
| facebookresearch/hydra | MIT | rebench_v2_field |
| facelessuser/soupsieve | MIT | rebench_v2_field |
| getmoto/moto | Apache-2.0 | rebench_v2_field |
| getnikola/nikola | MIT | github_api |
| iterative/dvc | Apache-2.0 | rebench_v2_field |
| jawah/charset_normalizer | MIT | github_api |
| john-kurkowski/tldextract | BSD-3-Clause | github_api |
| jsvine/pdfplumber | MIT | github_api |
| kennethreitz/records | ISC | github_api |
| marshmallow-code/apispec | MIT | rebench_v2_field |
| matthewwithanm/python-markdownify | MIT | rebench_v2_field |
| mido/mido | MIT | github_api |
| modin-project/modin | Apache-2.0 | rebench_v2_field |
| pallets/markupsafe | BSD-3-Clause | github_api |
| pandas-dev/pandas | BSD-3-Clause | rebench_v2_field |
| pdfminer/pdfminer.six | MIT | rebench_v2_field |
| pndurette/gTTS | MIT | github_api |
| pudo/dataset | MIT | github_api |
| rsalmei/alive-progress | MIT | github_api |
| rubik/radon | MIT | github_api |
| scrapy/scrapy | BSD-3-Clause | rebench_v2_field |
| seatgeek/thefuzz | MIT | github_api |
| sloria/environs | MIT | rebench_v2_field |
| sqlfluff/sqlfluff | MIT | github_api |
| stanfordnlp/dspy | MIT | rebench_v2_field |
| stanfordnlp/string2string | MIT | github_api |
| sunpy/sunpy | BSD-2-Clause | github_api |
| tkrajina/gpxpy | Apache-2.0 | github_api |
| tobymao/sqlglot | MIT | github_api |
| tox-dev/pipdeptree | MIT | rebench_v2_field |
| tweepy/tweepy | MIT | github_api |
| weaveworks/grafanalib | Apache-2.0 | github_api |

No repo outside A and B is considered. Run-001 repos (5 Train, 3 held-out) are never candidates.

## 2. Repository admission rules (all must hold)

R1 Licence. The survey SPDX id is in `sources.PERMISSIVE_LICENSES`. Before admission it is
re-checked with the code-host licence API (the same check as `scripts/learning_pipeline_v1.py
check-licenses`); a missing or different non-permissive answer excludes the repo. Repos whose survey
source is `rebench_v2_field` are admitted only if this API re-check returns a permissive SPDX id.
Group B additionally requires an explicit `REPO_LICENSES` code change (reviewed PR) before any build.

R2 Exposure (repo level). `exclusions.is_excluded(upstream, excluded_repositories(scan_vare_tree(main)))`
returns None, with the scan run on `origin/main` using the config's `scan_skip`, and
`check_required(scan)` returns []. Not a run-001 Train or held-out repository.

R3 Exposure (instance level). Every instance ID in the Train manifest, the run-001 held-out manifest
or the exposure ledger is removed. A repo with exposed IDs is NOT excluded wholesale for that reason
alone (applies to jd/tenacity: its 1 exposed task is removed, the repo stays a candidate).

R4 Metadata. >= 8 tasks with 1 <= |FAIL_TO_PASS| <= 30 and |PASS_TO_PASS| <= 400 after R3.

R5 Heavy-repo rule, part 1 (decided now; evaluated from packaging files only, never task content,
at the pinned mirror commit). The repo is excluded if either holds:
  (a) compiled build: `setup.py` contains `ext_modules`, `Extension(` or `cythonize`; or
      `pyproject.toml` `[build-system]` `build-backend` is one of `mesonpy`, `scikit_build_core.build`,
      `maturin`, or its `requires` names `cython`, `pybind11`, `setuptools-rust`, `numpy` or
      `extension-helpers` (case-insensitive);
  (b) ML-framework runtime dependency: declared runtime requirements (`[project] dependencies`,
      `install_requires` in `setup.py`/`setup.cfg`, or root `requirements.txt`) name `torch`,
      `tensorflow`, `jax`, `jaxlib`, `mxnet` or `paddlepaddle`.
  Content filters are still run on R5-excluded repos and reported for information only; they are not
  admissible.

R6 Content filters (unchanged builder code, see section 3): the repo needs >= k passers, where k is
the per-repo task count of the chosen design (k in {7, 8}).

R7 Environment buildability (judged LATER on the Mac, not now). On the Mac, with the same Python
interpreter and pipeline code used for run-001:
  (a) `RepoEnv.materialize()` exits without error (clone, HEAD == pinned commit, `pip install -e .`
      + `pytest==8.3.5` + any `EXTRA_TEST_DEPS`) within its 900 s install timeout, with no manual
      steps and no system packages beyond those already installed for run-001;
  (b) heavy-repo rule, part 2: resulting venv <= 1.0 GiB (`du -sk` of the venv) and clone <= 300 MB;
  (c) task validation, applied to candidate tasks in the sampling order of section 4: with the bug
      patch applied, at least one FAIL_TO_PASS test fails (bug-only grade fails), and with the
      reference file restored every FAIL_TO_PASS and PASS_TO_PASS test is PASSED (`RepoEnv.grade`
      with candidate_text = pinned original file). Tasks failing (c) are skipped and the next task in
      order is tried; the repo is admitted only if k tasks validate.
  Repos failing (a) or (b) are dropped as a whole. No dependency pins may be added after seeing
  validation results except by an addendum written before any model run.

## 3. Content filters (exactly the builder's, run on the box)

Pool: tasks of candidate repos that pass R3 and R4. Filters, in builder order, using the builder
functions unmodified:
1. `curriculum.prefilter(row, cfg)`: single-file patch; parseable; target is library `.py` (not a test
   path); <= 30 changed lines; F2P/P2P counts; problem statement non-empty and <= 3000 chars.
2. Duplicate: builder rule (normalized patch sha256, or (mirror, target file, problem-statement sha256)),
   processed in instance_id order over all prefilter-passing rows of the run-001 repos and all
   candidate repos (exposed rows included); additionally a non-exposed task that duplicates any exposed
   or manifest task is dropped regardless of order.
3. `curriculum._file_check(env, task, cfg)` with a `RepoEnv` at the mirror's pinned commit
   (= mirror default-branch HEAD from `remote_head`, recorded): bug patch applies (`git apply`), target
   file <= 8000 chars, reference blocks apply, and they restore the pinned file exactly. The checkout may
   be a partial/sparse clone containing only the needed paths (equivalent for single-file patches).
The prompt-length gate of the pool probe is not part of the builder and is not applied here.

Confidentiality: content is processed programmatically only. Only per-repo pass counts, per-filter drop
counts and passer instance IDs are saved. No problem statement, patch, test name or file content is
printed, logged or stored.

## 4. Sampling (seed fixed now)

SEED = `vare-run002-heldout-pool-v1`.
- Task order within a repo: ascending `curriculum.order_key(SEED, instance_id)`
  (= sha256 of `"{SEED}:{instance_id}"`), over content-filter passers. On the Mac, tasks are validated
  (R7c) in this order and the first k that validate are taken.
- Repo choice when more repos qualify than R: ascending sha256 of `"{SEED}:repo:{upstream}"` over
  repos that satisfy R1-R7; take the first R. Groups A and B are pooled with no preference.
- R and k are fixed from the verified counts and the power table before any model run, and are not
  changed after validation except by shrinking R when fewer repos survive R7 (reported as such).
