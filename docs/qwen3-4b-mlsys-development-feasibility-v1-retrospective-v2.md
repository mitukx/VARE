# Qwen3-4B Development feasibility v1 retrospective audit v2

## Scope

This append-only audit corrects the failure taxonomy for the retained v1 Qwen run. It reads the existing checksum-verified run and its historical implementation commit. It did not rerun model inference, reapply any patch, or rerun an independent grader. Original feasibility.json, raw-records.jsonl, and SHA256SUMS are unchanged.

## Reproduction

    PYTHONPATH=src python scripts/audit_qwen3_4b_mlsys_development_feasibility_v2.py --run results/qwen3-4b-mlsys-development-feasibility-v1/run-1 --output results/qwen3-4b-mlsys-development-feasibility-v1/retrospective-audit-v2.json

## Result

- Retained rollout records: 15.
- Independent verified successes: 0.
- Actual formatting failures: 13. These included Markdown fences and leading reasoning/code instead of one exact unified diff.
- Patch application failures: 2 (Transformers sampled seeds 7402 and 7403). Both passed the task helper's path-format validation, then git apply --check rejected the patch; the task helper returned “candidate patch did not apply cleanly”. The frozen runner incorrectly labeled these format_failure because the task helper exposed both parse and application errors as ValueError.
- Grader wrong answers, grader timeouts, infrastructure failures: 0 observed. The two application failures never reached the independent grader.
- Sampled task pass@4, binary reward variance, and independent reward remain unchanged at zero for all three Development tasks.

The machine-readable additive result is results/qwen3-4b-mlsys-development-feasibility-v1/retrospective-audit-v2.json (SHA-256 2375ff4441a7d06dfc5b802c4562bd7b703d6506685c74111c1d840cd72231f9).

## Prospective contract

vare.qwen3-patch-output-contract.retrospective-v2 validates exactly one full unified Git diff for the allowlisted target, including exact hunk line counts. It rejects leading/trailing prose, Markdown fences, extra files, and malformed or truncated hunks before apply/grading. It separates format failure, patch application failure/timeout, wrong answer, grader timeout/failure, and grader infrastructure failure.

The new runner wrapper checks its implementation lock and refuses the v1 task set before model loading because all three tasks were exposed in the v1 model run. The new contract is not permission to repeat those tasks; any fresh inference requires a disjoint cohort and a new protocol version. Focused contract tests cover valid in-hunk context, malformed outputs, failure phases, the protocol hash lock, and the consumed-task refusal.

## Limits

This corrects historical failure classification and adds a tested future contract. It produces no new model outcome and no capability result. The historical v1 study remains a NO-GO: zero verified successes and zero sampled reward variance.
