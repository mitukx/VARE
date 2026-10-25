# Transformers #47468 qualification v1

This is a frozen **pre-outcome** Development-task qualification. It contains no baseline, mutant, or known-fix test result and no model inference. The upstream issue, implementation and regression test are public; external model-training exposure is unknown. This task cannot be reported as held out.

## Reconstruct the pinned inputs

From the VARE checkout, prepare a local upstream Git cache before running the grader:

```bash
git init work/transformers-source-cache-v1
git -C work/transformers-source-cache-v1 remote add origin https://github.com/huggingface/transformers.git
git -C work/transformers-source-cache-v1 fetch --depth=1 --filter=blob:none origin 6a994bc42f4c2c3e53ca384aff84750b85de0e6e 3da1cdf80218c71ac4b4121cca43ea7def44e68a
git -C work/transformers-source-cache-v1 archive --format=tar 6a994bc42f4c2c3e53ca384aff84750b85de0e6e src/transformers > work/transformers-src-base.tar
git -C work/transformers-source-cache-v1 show 3da1cdf80218c71ac4b4121cca43ea7def44e68a:tests/utils/test_generic.py > work/transformers-grader-v1/tests/utils/test_generic.py
git -C work/transformers-source-cache-v1 show 3da1cdf80218c71ac4b4121cca43ea7def44e68a:src/transformers/utils/generic.py > work/transformers-47468-controls-v1/fix-generic.py
```

The qualification runner checks the archive, grader test, fix target, mutant and Python dependency freezes against `protocol-v1.json` and `protocol-v1.lock.json`. The base source archive contains the `src/transformers` subtree only. It excludes repository history, tests, bytecode, wheels, the PR patch and fix bytes. The regression test and fix are stored outside candidate-visible source and are used only by the trusted operator-side qualification runner.

## Environment and isolation

The frozen local preflight uses Python 3.12.12 on macOS arm64, the recorded Miniforge environment plus an already-installed Transformers agent environment overlay, PyTorch 2.9.1, NumPy 2.5.4, pytest 8.4.2, tokenizers 0.23.3 and huggingface-hub 2.2.0. Complete `pip freeze` outputs are in `locks/`. No packages, model weights or paid services were downloaded for this qualification.

The grader subprocess uses macOS `sandbox-exec` with network access denied. A sandboxed socket probe was denied with `EPERM`. This is a network boundary for a trusted upstream unit test, not a container or general sandbox for hostile code. The candidate model receives only the task prompt and the baseline target through allowlisted source tools; it has no shell, test-running tool, test file, fix or grader result.

## Frozen test and controls

The one selected public test is `GenericTester.test_torch_float` from the pinned upstream fix commit. The required sequence is baseline failure, a one-expression incorrect mutant failure, and exact-fix success. Only an assertion failure in the selected test counts as the two expected failures; import or collection errors do not. One invocation per phase is allowed. Any hash, resource, sandbox or unexpected-outcome failure stops qualification as `UNQUALIFIED`.

After independent review of the exact committed protocol and runner, `python scripts/run_transformers_47468_qualification_v1.py --freeze-record` records the exact source commit without running tests. Only after that review should `--execute` be considered. A failed or partial `run-v1` is immutable and the one-shot runner will not overwrite or repeat it.

## Claim boundary

A complete passing control sequence can certify at most one Development source/grader family. It does not establish that a model can solve the task, that the task is unseen in pretraining, that a policy improves, or that the result generalizes to other Transformers tasks.
