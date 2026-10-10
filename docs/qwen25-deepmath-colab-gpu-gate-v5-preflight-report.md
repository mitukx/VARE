# DeepMath GPU inference gate v5: preflight result

**Date:** 2026-10-11 (Asia/Tokyo)  
**Protocol:** `qwen25_deepmath_grpo_math500_v5`  
**Code commit:** `19dc779888798d5110a457983732af498318a815`

## Decision

**NO-GO before model inference.** The free Colab T4 and Python runtime met the v5 hardware/runtime eligibility checks, but the pinned dependency installation left `pip check` failing. The v5 runbook requires a clean dependency check and says to stop on failure. No model or TRAIN parquet was downloaded to Colab, no prompt was run, and no model inference or optimizer update occurred.

## Verified before the stop

- Colab selected a T4 GPU runtime with Python 3.12.13.
- PyTorch was `2.11.0+cu128` with CUDA build `12.8`; CUDA was available and one GPU was visible.
- The GPU reported `Tesla T4`, `15,637,086,208` total VRAM bytes, compute capability `(7, 5)`; driver `580.82.07` reported CUDA `13.0`.
- The frozen model file hashes and TRAIN parquet hash had already matched the v4/v5 lock on the local CPU preflight. The fixed TRAIN prompt fingerprint was `0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a`.
- The T4 reserved-memory ceiling is `min(14 GiB, 0.90 × 15,637,086,208) = 14,073,377,587` bytes. The v5 lock and regression test record this derived ceiling.
- The exact v5 non-Torch package pins installed successfully: Transformers 5.5.4, TRL 1.1.0, Accelerate 1.13.0, Datasets 4.8.4, math-verify 0.9.0, latex2sympy2-extended 1.11.0, antlr4-python3-runtime 4.13.2, and PyArrow 22.0.0. The existing CUDA Torch wheel remained installed.

## Failing dependency check

`pip check` reported:

```text
ipython 7.34.0 requires jedi, which is not installed.
omegaconf 2.3.1 has requirement antlr4-python3-runtime==4.9.*, but you have antlr4-python3-runtime 4.13.2.
```

The second conflict is between a Colab-preinstalled package and the frozen antlr pin. The first is another inconsistency in the preinstalled runtime. Neither was resolved by changing package pins or removing preinstalled packages. Under the frozen v5 stop condition, the runner was not launched.

## Artifacts and reproducibility

- Colab notebook: `https://colab.research.google.com/drive/1QPH7SITP9Fsn3lpmht9XQkob9QatQ9UT`
- Persistent Drive directory: `MyDrive/VARE/deepmath-gate-v5`
- Runtime and installation records: `pip-install-v5.log`, `pip-check-v5.log`; the notebook retains the initial hardware/package report and failed preflight cell output.
- No runner journal or generation output exists.
- The v5 protocol, runner, and tests are in the code commit above. CI runs and the final documentation commit are recorded in the repository history.

## Interpretation

This is an environment compatibility NO-GO, not a task-performance FAIL. There are no success-rate, format-rate, reward-variance, token-throughput, or model VRAM measurements because no model was loaded. The v5 one-shot gate is spent as a failed preflight; retrying it would violate the frozen no-rescue rule. Any future attempt needs a separately approved protocol revision that resolves the runtime conflict without silently changing the frozen study conditions.
