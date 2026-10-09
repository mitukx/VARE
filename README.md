# VARE

VARE is a CPU-first research project for studying post-training signals, policy updates, and evaluation reliability under limited compute. It uses frozen protocols, independent checks, provenance tracking, and retained negative results.

## Current status

VARE has **not demonstrated an independently confirmed improvement in real-model task success after post-training**. Recent small-model comparisons have failed their predeclared gates; see [current evidence gaps](docs/current-gaps.md) and the [experiment index](docs/experiments.md).

The current review artifact is a narrow follow-up patch for RVL rollback behavior when a model contains a shared child module. Its regression fails on the pinned merged RVL revision and passes with the patch. It is reproducible locally, but has **not** been submitted upstream or independently reviewed by a human. Production incidence and model-quality impact are unknown.

## Review this project

1. Read the [technical reviewer guide](docs/reviewer-guide.md) for the review path, reproduction command, code map, and claim limits.
2. Read the [shared-module rollback report](docs/rvl-module-mode-shared-alias-followup-v1.md) and inspect the [proposed patch](contributions/rvl-module-mode-shared-alias-followup.patch).
3. Review the [retained reproduction evidence](results/rvl-module-mode-shared-alias-v1/manifest.json), including the pinned base, patch hashes, baseline failure, and patched test result.
4. Read the [OpenBookQA GRPO/SFT confirmation report](docs/openbookqa-qwen25-grpo-sft-confirmation-v1-report.md) for the latest completed real-model comparison and its negative result.
5. Use the [evidence ledger](docs/evidence.md), [current gaps](docs/current-gaps.md), and [roadmap](docs/roadmap.md) for broader context. The [experiment index](docs/experiments.md) retains older studies and failures.

## Selected results

| Area | Evidence | Limit |
| --- | --- | --- |
| RVL rollback | On pinned RVL `27ebf7f`, the shared-module regression fails on exact mode restoration and the subsequent Dropout forward. The proposed correction passes all 14 `tests.test_mini_lab_torch` tests in a clean clone. | Deliberately constructed shared-module graph; production incidence, CUDA/distributed behavior, and model-quality impact are unproven. No upstream submission or independent human review. |
| OpenBookQA post-training | On 128 confirmation items, base exact-answer success was 45.31%, matched SFT averaged 33.33%, and GRPO scored 0/128 for each of three seeds because all responses failed the frozen answer parser. Independent reconstruction passed. | One small model/task/protocol; the result retires that pairing and does not show that GRPO generally fails. The dataset license is listed as unknown. |
| Earlier ARC comparison | Three-seed GRPO averaged 38.84% exact success versus 40.00% for matched SFT and missed its frozen advancement gate. | Negative small-model evidence, not a general method comparison. |

## Repository map

| Path | Contents |
| --- | --- |
| `src/vare/`, `tests/` | VARE implementation and regression tests |
| `protocols/` | Frozen study definitions, gates, and hashes |
| `results/` | Retained outputs, manifests, and failed or passed runs |
| `docs/` | Reports, evidence ledger, decisions, and review guide |
| `contributions/` | Pinned upstream patch artifacts and their status |
| `scripts/`, `reproducers/` | Reproduction, audit, and diagnostic commands |

To reproduce the current RVL follow-up on CPU, from the repository root:

```bash
PYTHON=python3.12 bash scripts/reproduce_rvl_shared_module_mode_followup_v1.sh \
  /tmp/vare-rvl-shared-mode-review
```

The script needs Git and network access to clone the pinned RVL revision, plus Python 3.12, PyTorch 2.9.1, and Transformers 4.57.3. It runs the focused regression on an unpatched clone and all 14 trainer tests on a patched clone.

Large checkpoints and benchmark content with uncertain redistribution terms remain outside version control. Protocol locks retain source revisions and hashes where applicable. VARE's claims are limited to the exact tasks, models, code paths, and checks described in each report; synthetic or correctness results are not model-capability results.

## License

Apache-2.0. See [LICENSE](LICENSE).
