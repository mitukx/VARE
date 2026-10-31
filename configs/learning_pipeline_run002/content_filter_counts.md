# Run-002 held-out pool: builder content-filter counts (verified, counts only)

Criteria: `allowlist_extension_criteria.md`, sha256 `7ed490f8945f9ad395d3f5752540aa5e5148aca6cd99c6d541e3f6ba8e8f3cab` (frozen before this run).
Pipeline code/config: VARE `6cafa37a` (unmodified builder functions). Dataset rev `77cab905`. Filter values: {'max_changed_lines': 30, 'max_fail_to_pass': 30, 'max_pass_to_pass': 400, 'max_problem_chars': 3000, 'max_file_chars': 8000}.
Exposure: 263 IDs removed (ledger 263 ⊇ Train+run-001 held-out manifests 122). Exclusion scan of the pinned tree: 7598 files, 37 identifiers, required identifiers missing: []; none of the 69 candidates excluded.
Download: 11.4 MB of git objects (blob-less depth-1 clones, sparse checkout of target + packaging files) plus ~70 small licence-API responses and ls-remote calls. Shards were already cached. No repo was skipped for size; **all 69 evaluated**.
Cross-check: for the 24 allowlisted repos, _file_check on a copy of the cached full clone agreed with the sparse checkout on 1070/1070 tasks (HEAD matched in all 24).

## Summary

* Metadata-eligible, disjoint tasks: 13570; after static filters + dedup: 8550; after file checks (**passers**): 1471.
* Drops by filter (all 69 repos): not_single_file 1580, unsupported_patch 3, target_not_library_python 288, patch_too_large 1873, problem_statement_length 1270, duplicate 6, duplicate_of_exposed 0, bug_patch_does_not_apply 0, target_file_too_long 6692, reference_blocks_do_not_apply 1, reference_blocks_do_not_restore_file 386.
* Repos admissible before the Mac environment stage (R1–R5 hold) with **≥8 passers: 35**, **≥7 passers: 36**. Including the 7 repos excluded by the heavy-repo rule R5: ≥8 38, ≥7 39.
  * group A: ≥8 9, ≥7 10
  * group B: ≥8 26, ≥7 26
* Licence re-check (code-host licence API, today): all 69 return the survey's SPDX id; all permissive.
* R5 heavy-repo exclusions: Project-MONAI/MONAI (1 passers: setup.py:ext_modules, runtime_deps:torch); amueller/word_cloud (9 passers: setup.py:ext_modules, setup.py:cythonize, pyproject:requires:cython); jawah/charset_normalizer (1 passers: setup.py:ext_modules); pallets/markupsafe (1 passers: setup.py:ext_modules, setup.py:Extension(); pandas-dev/pandas (17 passers: setup.py:ext_modules, setup.py:Extension(, setup.py:cythonize, pyproject:build-backend, pyproject:requires:cython, pyproject:requires:numpy); stanfordnlp/string2string (6 passers: runtime_deps:torch); sunpy/sunpy (113 passers: pyproject:requires:extension-helpers, pyproject:requires:numpy).
* These are upper bounds for the environment stage: R7 (Mac install, venv ≤1 GiB, bug-fails/reference-passes validation) is not yet applied.

## Per repo

| repo | grp | licence (survey src) | meta-eligible | static pass | **passers** | multi | unsup | test/non-py | >30 lines | PS len | dup | dup-exp | no-apply | >8000 | ref-no-apply | no-restore | R5 heavy | admissible pre-env |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cantools/cantools | B | MIT (github_api) | 1184 | 700 | **173** | 88 | 0 | 36 | 151 | 209 | 0 | 0 | 0 | 519 | 0 | 8 |  | yes |
| sqlfluff/sqlfluff | B | MIT (github_api) | 1157 | 800 | **122** | 111 | 0 | 0 | 223 | 20 | 3 | 0 | 0 | 647 | 0 | 31 |  | yes |
| stanfordnlp/dspy | B | MIT (rebench_v2_field) | 565 | 432 | **122** | 34 | 0 | 0 | 99 | 0 | 0 | 0 | 0 | 273 | 0 | 37 |  | yes |
| tox-dev/pipdeptree | B | MIT (rebench_v2_field) | 238 | 169 | **116** | 34 | 0 | 0 | 35 | 0 | 0 | 0 | 0 | 52 | 0 | 1 |  | yes |
| sunpy/sunpy | B | BSD-2-Clause (github_api) | 709 | 505 | **113** | 96 | 0 | 0 | 104 | 3 | 1 | 0 | 0 | 379 | 0 | 13 | yes | no |
| mido/mido | B | MIT (github_api) | 318 | 146 | **68** | 77 | 0 | 0 | 12 | 83 | 0 | 0 | 0 | 69 | 0 | 9 |  | yes |
| iterative/dvc | B | Apache-2.0 (rebench_v2_field) | 407 | 186 | **51** | 134 | 0 | 0 | 83 | 4 | 0 | 0 | 0 | 96 | 0 | 39 |  | yes |
| jd/tenacity | A | Apache-2.0 (github_api) | 123 | 106 | **45** | 9 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 58 | 0 | 3 |  | yes |
| scrapy/scrapy | B | BSD-3-Clause (rebench_v2_field) | 146 | 90 | **41** | 35 | 0 | 0 | 20 | 1 | 0 | 0 | 0 | 34 | 0 | 15 |  | yes |
| pdfminer/pdfminer.six | B | MIT (rebench_v2_field) | 488 | 411 | **36** | 1 | 0 | 0 | 69 | 7 | 0 | 0 | 0 | 369 | 0 | 6 |  | yes |
| cknd/stackprinter | A | MIT (github_api) | 137 | 103 | **35** | 0 | 0 | 0 | 34 | 0 | 0 | 0 | 0 | 67 | 0 | 1 |  | yes |
| rsalmei/alive-progress | B | MIT (github_api) | 139 | 95 | **33** | 16 | 0 | 0 | 13 | 15 | 0 | 0 | 0 | 61 | 0 | 1 |  | yes |
| theskumar/python-dotenv | A | BSD-3-Clause (rebench_v2_field) | 106 | 59 | **33** | 41 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 26 | 0 | 0 |  | yes |
| tobymao/sqlglot | B | MIT (github_api) | 559 | 424 | **33** | 42 | 0 | 0 | 86 | 7 | 0 | 0 | 0 | 383 | 0 | 8 |  | yes |
| django/channels | B | BSD-3-Clause (rebench_v2_field) | 100 | 89 | **32** | 2 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 57 | 0 | 0 |  | yes |
| facelessuser/soupsieve | B | MIT (rebench_v2_field) | 189 | 138 | **29** | 0 | 0 | 0 | 25 | 26 | 0 | 0 | 0 | 105 | 0 | 4 |  | yes |
| rubik/radon | B | MIT (github_api) | 201 | 126 | **27** | 29 | 0 | 0 | 23 | 23 | 0 | 0 | 0 | 93 | 0 | 6 |  | yes |
| weaveworks/grafanalib | B | Apache-2.0 (github_api) | 118 | 111 | **25** | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 86 | 0 | 0 |  | yes |
| alanjds/drf-nested-routers | B | Apache-2.0 (github_api) | 35 | 26 | **24** | 0 | 0 | 0 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |  | yes |
| tkrajina/gpxpy | B | Apache-2.0 (github_api) | 643 | 360 | **23** | 0 | 0 | 246 | 33 | 4 | 0 | 0 | 0 | 330 | 0 | 7 |  | yes |
| getmoto/moto | B | Apache-2.0 (rebench_v2_field) | 481 | 251 | **20** | 156 | 0 | 0 | 48 | 26 | 0 | 0 | 0 | 216 | 0 | 15 |  | yes |
| mewwts/addict | A | MIT (github_api) | 20 | 20 | **20** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | yes |
| jsvine/pdfplumber | B | MIT (github_api) | 226 | 170 | **19** | 0 | 0 | 0 | 56 | 0 | 0 | 0 | 0 | 143 | 0 | 8 |  | yes |
| conan-io/conan | B | MIT (rebench_v2_field) | 877 | 423 | **17** | 343 | 3 | 0 | 97 | 10 | 1 | 0 | 0 | 296 | 1 | 109 |  | yes |
| cool-RR/PySnooper | A | MIT (github_api) | 41 | 26 | **17** | 0 | 0 | 0 | 2 | 13 | 0 | 0 | 0 | 9 | 0 | 0 |  | yes |
| pandas-dev/pandas | B | BSD-3-Clause (rebench_v2_field) | 690 | 381 | **17** | 58 | 0 | 3 | 82 | 166 | 0 | 0 | 0 | 356 | 0 | 8 | yes | no |
| pudo/dataset | B | MIT (github_api) | 101 | 41 | **17** | 0 | 0 | 0 | 12 | 48 | 0 | 0 | 0 | 21 | 0 | 3 |  | yes |
| alecthomas/voluptuous | A | BSD-3-Clause (github_api) | 218 | 168 | **16** | 0 | 0 | 0 | 24 | 26 | 0 | 0 | 0 | 152 | 0 | 0 |  | yes |
| erikrose/parsimonious | A | MIT (github_api) | 95 | 89 | **16** | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 73 | 0 | 0 |  | yes |
| john-kurkowski/tldextract | B | BSD-3-Clause (github_api) | 67 | 26 | **13** | 3 | 0 | 0 | 12 | 26 | 0 | 0 | 0 | 10 | 0 | 3 |  | yes |
| pndurette/gTTS | B | MIT (github_api) | 45 | 30 | **13** | 7 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 15 | 0 | 2 |  | yes |
| facebookresearch/fvcore | B | Apache-2.0 (github_api) | 445 | 43 | **10** | 81 | 0 | 0 | 35 | 286 | 0 | 0 | 0 | 26 | 0 | 7 |  | yes |
| seatgeek/thefuzz | B | MIT (github_api) | 32 | 29 | **10** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 19 | 0 | 0 |  | yes |
| amueller/word_cloud | B | MIT (github_api) | 63 | 45 | **9** | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 33 | 0 | 3 | yes | no |
| hukkin/tomli | A | MIT (github_api) | 80 | 45 | **9** | 26 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 36 | 0 | 0 |  | yes |
| spulec/freezegun | A | Apache-2.0 (github_api) | 85 | 80 | **9** | 0 | 0 | 0 | 3 | 1 | 1 | 0 | 0 | 71 | 0 | 0 |  | yes |
| cookiecutter/cookiecutter | B | BSD-3-Clause (github_api) | 48 | 23 | **8** | 0 | 0 | 0 | 25 | 0 | 0 | 0 | 0 | 9 | 0 | 6 |  | yes |
| tweepy/tweepy | B | MIT (github_api) | 255 | 234 | **8** | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | 226 | 0 | 0 |  | yes |
| un33k/python-slugify | A | MIT (github_api) | 16 | 10 | **7** | 2 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |  | yes |
| life4/textdistance | A | MIT (github_api) | 12 | 8 | **6** | 1 | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |  | yes |
| stanfordnlp/string2string | B | MIT (github_api) | 128 | 117 | **6** | 0 | 0 | 0 | 11 | 0 | 0 | 0 | 0 | 111 | 0 | 0 | yes | no |
| dask/dask | B | BSD-3-Clause (rebench_v2_field) | 130 | 39 | **5** | 11 | 0 | 0 | 49 | 31 | 0 | 0 | 0 | 31 | 0 | 3 |  | yes |
| facebookresearch/hydra | B | MIT (rebench_v2_field) | 77 | 33 | **5** | 19 | 0 | 0 | 24 | 1 | 0 | 0 | 0 | 19 | 0 | 9 |  | yes |
| django/daphne | B | BSD-3-Clause (github_api) | 50 | 30 | **4** | 0 | 0 | 3 | 17 | 0 | 0 | 0 | 0 | 23 | 0 | 3 |  | yes |
| modin-project/modin | B | Apache-2.0 (rebench_v2_field) | 87 | 59 | **2** | 10 | 0 | 0 | 14 | 4 | 0 | 0 | 0 | 52 | 0 | 5 |  | yes |
| termcolor/termcolor | A | MIT (github_api) | 9 | 2 | **2** | 2 | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |  | yes |
| Project-MONAI/MONAI | B | Apache-2.0 (github_api) | 404 | 56 | **1** | 102 | 0 | 0 | 71 | 175 | 0 | 0 | 0 | 49 | 0 | 6 | yes | no |
| burnash/gspread | B | MIT (github_api) | 85 | 60 | **1** | 0 | 0 | 0 | 24 | 1 | 0 | 0 | 0 | 59 | 0 | 0 |  | yes |
| getnikola/nikola | B | MIT (github_api) | 62 | 49 | **1** | 1 | 0 | 0 | 10 | 2 | 0 | 0 | 0 | 47 | 0 | 1 |  | yes |
| jawah/charset_normalizer | B | MIT (github_api) | 115 | 96 | **1** | 2 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 94 | 0 | 1 | yes | no |
| pallets/markupsafe | B | BSD-3-Clause (github_api) | 30 | 26 | **1** | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 25 | 0 | 0 | yes | no |
| arrow-py/arrow | A | Apache-2.0 (github_api) | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | yes |
| bottlepy/bottle | B | MIT (github_api) | 315 | 280 | **0** | 0 | 0 | 0 | 29 | 6 | 0 | 0 | 0 | 280 | 0 | 0 |  | yes |
| chardet/chardet | A | 0BSD (github_api) | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | yes |
| datamade/usaddress | B | MIT (github_api) | 8 | 5 | **0** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |  | yes |
| dbader/schedule | A | MIT (github_api) | 63 | 55 | **0** | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 55 | 0 | 0 |  | yes |
| google/textfsm | A | Apache-2.0 (github_api) | 78 | 66 | **0** | 0 | 0 | 0 | 11 | 1 | 0 | 0 | 0 | 66 | 0 | 0 |  | yes |
| gruns/icecream | A | MIT (github_api) | 9 | 6 | **0** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |  | yes |
| keleshev/schema | A | MIT (github_api) | 56 | 48 | **0** | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 0 | 48 | 0 | 0 |  | yes |
| kennethreitz/records | B | ISC (github_api) | 34 | 28 | **0** | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 0 | 28 | 0 | 0 |  | yes |
| lepture/mistune | A | BSD-3-Clause (github_api) | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  | yes |
| madzak/python-json-logger | A | BSD-2-Clause (github_api) | 38 | 33 | **0** | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 33 | 0 | 0 |  | yes |
| marshmallow-code/apispec | B | MIT (rebench_v2_field) | 10 | 4 | **0** | 2 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |  | yes |
| martinblech/xmltodict | A | MIT (github_api) | 31 | 14 | **0** | 1 | 0 | 0 | 4 | 12 | 0 | 0 | 0 | 14 | 0 | 0 |  | yes |
| matthewwithanm/python-markdownify | B | MIT (rebench_v2_field) | 73 | 50 | **0** | 1 | 0 | 0 | 12 | 10 | 0 | 0 | 0 | 50 | 0 | 0 |  | yes |
| r1chardj0n3s/parse | A | MIT (github_api) | 58 | 44 | **0** | 0 | 0 | 0 | 9 | 5 | 0 | 0 | 0 | 44 | 0 | 0 |  | yes |
| rustedpy/result | A | MIT (github_api) | 59 | 51 | **0** | 0 | 0 | 0 | 1 | 7 | 0 | 0 | 0 | 51 | 0 | 0 |  | yes |
| sloria/environs | B | MIT (rebench_v2_field) | 56 | 44 | **0** | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 44 | 0 | 0 |  | yes |
| vi3k6i5/flashtext | A | MIT (github_api) | 46 | 37 | **0** | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 37 | 0 | 0 |  | yes |

(`jd/tenacity`: 1 exposed task removed at instance level per R3.)

## Power (power_designs.py model, balanced R×k feasible under the verified counts)

Sign-flip, α=0.025 (Holm plan), 1000 reps per cell (MC SE ≈0.013 at 0.8), loss 0.05, seed SD 0.25, σ_w proxy 1.868, 3 seeds. Feasible R: k=7 up to 36, k=8 up to 35.

| design | N | ICC .05 lift .05 | .075 | .10 | ICC .10 lift .05 | .075 | .10 |
|---|---|---|---|---|---|---|---|
| 14×7 | 98 | 0.33 | 0.60 | 0.77 | 0.27 | 0.51 | 0.70 |
| 15×7 | 105 | 0.38 | 0.64 | 0.81 | 0.32 | 0.56 | 0.76 |
| 14×8 | 112 | 0.39 | 0.63 | 0.82 | 0.32 | 0.54 | 0.74 |
| 16×7 | 112 | 0.40 | 0.67 | 0.85 | 0.36 | 0.60 | 0.81 |
| 17×7 | 119 | 0.45 | 0.70 | 0.89 | 0.39 | 0.61 | 0.80 |
| 15×8 | 120 | 0.44 | 0.68 | 0.85 | 0.35 | 0.61 | 0.78 |
| 18×7 | 126 | 0.47 | 0.75 | 0.93 | 0.42 | 0.66 | 0.84 |
| 16×8 | 128 | 0.46 | 0.74 | 0.90 | 0.42 | 0.63 | 0.83 |
| 19×7 | 133 | 0.50 | 0.78 | 0.93 | 0.45 | 0.71 | 0.87 |
| 17×8 | 136 | 0.49 | 0.78 | 0.91 | 0.42 | 0.69 | 0.85 |
| 20×7 | 140 | 0.54 | 0.83 | 0.92 | 0.47 | 0.71 | 0.90 |
| 18×8 | 144 | 0.52 | 0.80 | 0.93 | 0.46 | 0.73 | 0.90 |
| 21×7 | 147 | 0.56 | 0.83 | 0.95 | 0.50 | 0.79 | 0.91 |
| 19×8 | 152 | 0.54 | 0.84 | 0.95 | 0.49 | 0.75 | 0.92 |
| 22×7 | 154 | 0.58 | 0.84 | 0.97 | 0.54 | 0.80 | 0.92 |
| 20×8 | 160 | 0.58 | 0.83 | 0.96 | 0.51 | 0.79 | 0.92 |
| 23×7 | 161 | 0.63 | 0.88 | 0.96 | 0.55 | 0.81 | 0.94 |
| 21×8 | 168 | 0.62 | 0.86 | 0.96 | 0.56 | 0.82 | 0.93 |
| 24×7 | 168 | 0.61 | 0.89 | 0.97 | 0.56 | 0.84 | 0.95 |
| 25×7 | 175 | 0.64 | 0.89 | 0.97 | 0.60 | 0.85 | 0.95 |
| 22×8 | 176 | 0.63 | 0.89 | 0.98 | 0.59 | 0.83 | 0.95 |
| 26×7 | 182 | 0.69 | 0.92 | 0.98 | 0.62 | 0.87 | 0.96 |
| 23×8 | 184 | 0.69 | 0.90 | 0.98 | 0.59 | 0.84 | 0.94 |
| 27×7 | 189 | 0.71 | 0.93 | 0.98 | 0.64 | 0.88 | 0.97 |
| 24×8 | 192 | 0.68 | 0.91 | 0.99 | 0.63 | 0.87 | 0.97 |
| 28×7 | 196 | 0.72 | 0.94 | 0.98 | 0.68 | 0.88 | 0.97 |
| 25×8 | 200 | 0.73 | 0.92 | 0.98 | 0.65 | 0.87 | 0.96 |
| 29×7 | 203 | 0.76 | 0.94 | 0.99 | 0.69 | 0.90 | 0.97 |
| 26×8 | 208 | 0.72 | 0.93 | 0.98 | 0.67 | 0.90 | 0.97 |
| 30×7 | 210 | 0.76 | 0.94 | 0.99 | 0.70 | 0.91 | 0.97 |
| 27×8 | 216 | 0.77 | 0.94 | 0.99 | 0.68 | 0.90 | 0.98 |
| 31×7 | 217 | 0.78 | 0.96 | 0.99 | 0.71 | 0.92 | 0.99 |
| 28×8 | 224 | 0.77 | 0.94 | 0.99 | 0.69 | 0.91 | 0.99 |
| 32×7 | 224 | 0.79 | 0.95 | 1.00 | 0.69 | 0.94 | 0.99 |
| 33×7 | 231 | 0.83 | 0.96 | 1.00 | 0.74 | 0.94 | 0.99 |
| 29×8 | 232 | 0.79 | 0.96 | 1.00 | 0.73 | 0.92 | 0.99 |
| 34×7 | 238 | 0.81 | 0.96 | 0.99 | 0.73 | 0.93 | 0.99 |
| 30×8 | 240 | 0.80 | 0.96 | 1.00 | 0.73 | 0.94 | 0.98 |
| 35×7 | 245 | 0.84 | 0.98 | 1.00 | 0.79 | 0.95 | 0.99 |
| 31×8 | 248 | 0.83 | 0.98 | 1.00 | 0.77 | 0.94 | 0.99 |
| 36×7 | 252 | 0.84 | 0.97 | 1.00 | 0.80 | 0.95 | 1.00 |
| 32×8 | 256 | 0.85 | 0.97 | 1.00 | 0.76 | 0.94 | 1.00 |
| 33×8 | 264 | 0.86 | 0.98 | 1.00 | 0.80 | 0.96 | 0.99 |
| 34×8 | 272 | 0.87 | 0.98 | 1.00 | 0.81 | 0.96 | 0.99 |
| 35×8 | 280 | 0.85 | 0.99 | 1.00 | 0.82 | 0.96 | 1.00 |

Smallest design (by N) with ≥0.80 at lift 0.075: ICC 0.05 → **20x7** (N=140); ICC 0.10 → **23x7** (N=161).
