# Learning-pipeline v1: redacted public copies

Some committed learning-pipeline v1 evidence files record absolute paths of the hosts that produced
them. The originals are hash-bound (`run-001/SHA256SUMS.txt`, protocol and calibration records)
and stay unchanged. This directory holds derived public copies in which those host paths are
replaced by placeholders. Each copy keeps the original's repository-relative path under
`results/learning-pipeline-v1/public-redacted/`.

The copies are derivatives, not evidence. Hash checks, re-analysis and audits should use the
originals. Use a copy only where a file is quoted or redistributed.

## Redaction rule

The copies are produced by `scripts/redact_learning_pipeline_v1_public.py`.

* An ordered list of regular-expression substitutions is applied to the UTF-8 text of each
  original. Each rule runs on the output of the previous one, so the result is deterministic and
  idempotent.
* Path components are matched generically. Neither the script nor `MAPPING.json` spells a host
  directory name. The exact patterns are in `MAPPING.json` under `rule.substitutions`.
* Only the path prefix is replaced. The rest of each path, such as `<RUN_ROOT>/adapters/rft_self`,
  is kept, because it is the run-relative layout.
* JSON and JSON Lines copies must parse to the same structure as the original once the same
  substitution is applied to every string. The terminal logs are rich-text tables, so their
  column alignment shifts by the length difference.
* After substitution, a copy must not match the host-path pattern in
  `rule.verification_pattern`. The pattern covers macOS and Linux home, temporary, volume, mount
  and workspace roots, and Windows user directories. If any such string remains, generation
  fails rather than leaving it in place.

| Placeholder | Meaning |
|---|---|
| `<DATA_CACHE>` | dataset cache directory on the data-preparation host |
| `<HF_HUB_CACHE>` | Hugging Face hub cache directory on the training host |
| `<REPO_ROOT>` | repository checkout on the training host |
| `<RUN_ROOT>` | run-001 output directory on the training host |

## Verification

```
python scripts/redact_learning_pipeline_v1_public.py --check
pytest tests/test_redact_learning_pipeline_v1_public.py
```

`--check` fails (exit code 1) if any of the following holds:

* An original no longer has the sha256 recorded below, or no longer matches
  `run-001/SHA256SUMS.txt`.
* A copy differs from its regenerated form.
* A copy contains a host path.
* Any other file under `results/learning-pipeline-v1/` or `configs/learning_pipeline_v1/`
  contains a host path but has no copy.

`--write` regenerates the copies and `MAPPING.json`. It refuses to write while any check fails.

## Mapping

The same data, with byte sizes, is in `MAPPING.json`.

| Original | Original sha256 | Redacted copy | Redacted sha256 | Replacements |
|---|---|---|---|---|
| `configs/learning_pipeline_v1/build_report.json` | `3325f33af26d153c41cb30ec206778ce56a9197667fdaa6fdec791820b4ca704` | `results/learning-pipeline-v1/public-redacted/configs/learning_pipeline_v1/build_report.json` | `2381929bf4cf1506f0d571a70075238e08c2637a7e5db92f255c469e710d11b0` | <DATA_CACHE> x11 |
| `results/learning-pipeline-v1/run-001/adapters/rft_self/TRAINING_COMPLETE.json` | `f09035a7972bec491d421bbf4586668e7bd60658665ad143a341ad3002c8b667` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/rft_self/TRAINING_COMPLETE.json` | `7b004b4dc410e19523a8cfee6799f30bde47ecee517ca9cd3293bd9ccde99e9c` | <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/adapters/rft_self/adapter_config.json` | `12bc3e1f589a8d8092830b0f1be3a57f28cffa86cb99055d0785d82aaae0a3d7` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/rft_self/adapter_config.json` | `b6b4129f5a258a0a6604f62f2f1b2ec2d601d1ef0c510edca2175a74ebd88c20` | <HF_HUB_CACHE> x1, <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/adapters/rft_self/lora_config.json` | `2e08f036ff9f5a9520d32e29f61058cffe8d08b853ba12aa5adcea2ea2b637c8` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/rft_self/lora_config.json` | `0ae75bd498c905733ce4cdb9d614b0a3fefbcee442fc31c3a6f0c82f58cff8ec` | <HF_HUB_CACHE> x1, <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/adapters/sft_reference/TRAINING_COMPLETE.json` | `c6835c1c78ddd883d8612a11f4fbd48c2c82d10534d2f806506963ba96c82c10` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/sft_reference/TRAINING_COMPLETE.json` | `7740d1cb300f6c44c5c8ae438f9cc491f8487a5c543360426af7e404109e9aa6` | <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/adapters/sft_reference/adapter_config.json` | `b9011232389bdda388efdd8692060f99648f7338cc1a10343a4c5a0749b5c82d` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/sft_reference/adapter_config.json` | `f8a87a2edabbbc0ca5f4dc5f54b03cd5ad825b07e96f01affea804d2cd609b46` | <HF_HUB_CACHE> x1, <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/adapters/sft_reference/lora_config.json` | `d4dbee47d1f0ff9a5be0d155c8b85afaadca30d91df17f7913d5f1c4e6be20c0` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/adapters/sft_reference/lora_config.json` | `2b5b6c8e07bccaab5a8c8049a2738c2a08d275c331dae5200a2aff7e1d769ed8` | <HF_HUB_CACHE> x1, <RUN_ROOT> x2 |
| `results/learning-pipeline-v1/run-001/eval/no_update/graded.jsonl` | `afb5d597b1f2c832163de69362b1fda0b1fb35c23fa8024cd0f5923ab5f3ae00` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/eval/no_update/graded.jsonl` | `a17e7ec239830794296907a2337d0658dd542e917ae40fc4dd55bcad6d1dd953` | <HF_HUB_CACHE> x42 |
| `results/learning-pipeline-v1/run-001/eval/rft_self/graded.jsonl` | `061c9eae86dc52859646ca02abd69e73348c1c8146f18a6032739df3e67e0fe0` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/eval/rft_self/graded.jsonl` | `edb639881cd65b32aebdcd8fe6afd593a8869ebd6825d2420b2911950123a471` | <HF_HUB_CACHE> x42, <RUN_ROOT> x42 |
| `results/learning-pipeline-v1/run-001/eval/sft_reference/graded.jsonl` | `167d2937e8622206176228e8dc7042c96e16355b9b4496512b027190268e480b` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/eval/sft_reference/graded.jsonl` | `ad4463ade8f40b6390417eac07a8fadbef10778a31ae522541c5aad129b654c2` | <HF_HUB_CACHE> x42, <RUN_ROOT> x42 |
| `results/learning-pipeline-v1/run-001/events.jsonl` | `3d77239ff496ce5a258addbef109f1cb5848613b11130e3f8110e574024d6ca0` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/events.jsonl` | `174da0244b70a097cad4676765d4debd0909b242f34d76428061fdf9c3fd9e5d` | <RUN_ROOT> x4 |
| `results/learning-pipeline-v1/run-001/preflight_train_smoke/train.log` | `656247b583b886f7cc53e64e6c6dfa8035bd92c8e801aaeadc69c586c53f86df` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/preflight_train_smoke/train.log` | `0a9c6d3ca8bfabf1cb95fec812902a75867f6a911c86ce2746d1eaa39adff5fc` | <HF_HUB_CACHE> x1, <RUN_ROOT> x1 |
| `results/learning-pipeline-v1/run-001/run_meta.json` | `ac8593188d2d4690bb5077ba3d83142e728656c870df045639f78e63ade1d172` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/run_meta.json` | `ce1f5633bae70fb787f5fad485f377460830028c9fa9b5987a9210ab6acead8c` | <REPO_ROOT> x1 |
| `results/learning-pipeline-v1/run-001/train_report.json` | `c3e1b5f24c22574a5075323aacbb62d935783b44e945b5877b44c90ed3251883` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/train_report.json` | `8f1aafe649331c035dbba83c694af3f6aaa51c6f212e1dba112a84fc8af11da8` | <RUN_ROOT> x4 |
| `results/learning-pipeline-v1/run-001/train_rft_self.log` | `92f42a31b762916c05d8854d84138ad5548a98a85d444dd64cfb28da41c00b16` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/train_rft_self.log` | `e8a9f0f205f6761957483acae34851b4eb4c63dfb3eacc5413bfb24b5419f346` | <HF_HUB_CACHE> x1, <RUN_ROOT> x1 |
| `results/learning-pipeline-v1/run-001/train_sft_reference.log` | `90997aefd8222d4171cb6073599efd5645054640a28f7ef7cc0a2f831b88601c` | `results/learning-pipeline-v1/public-redacted/results/learning-pipeline-v1/run-001/train_sft_reference.log` | `bbf82335ec02560d4c45c550970a9f8221adcfc2a3d94caf97a48f7c89b28dc7` | <HF_HUB_CACHE> x1, <RUN_ROOT> x1 |

## Scope

The table covers every file under `results/learning-pipeline-v1/` and
`configs/learning_pipeline_v1/` that contains an absolute host path. Other historical result trees
in the repository also contain host paths. They are outside this directory's scope and are left
unchanged.
