# VARE Sprint X: shared-prefix MLX inference audit

**実施日:** 2026-10-28

**実験プロトコル:** `vare_sprint_x_schema_slots_kv_cache_v2`

**コード:** VARE `18afc60910ceb3ef5572fb2c7eadeff551bf9510`（PR #139、Draft）
**主張範囲:** 合成ツール対話における推論・prefill効率のみ。外部リポジトリ修復や一般的なCoding能力の評価ではない。

## 結果の要約

Qwen2.5-Coder-7B-Instruct-4bitをMLX Metalで一度ロードし、schema-slot controller v9とv10をABBA順で2組比較した。各attemptで`read_lines`、実際のhash-bound fixture読取り、`finish`を行った。v9/v10は2組とも生のtool actionと生成token IDが一致し、8/8 tool callsがstrict parserに受理された。4/4 attemptが2-tool-turnを完了した。

v10は各turnの共通prefixを一度prefillしてKV cacheを複製・再利用した。合計prefill対象tokenはattemptあたり18,669から6,637へ64.449%減り、ABBAの2組でwall timeは59.004%と61.306%短縮した。v10はこの固定された合成promptで再現可能な高速化を示したが、タスク成功率やモデル能力の改善は示していない。

## 事前固定と再現性

- 仮説、ABBA順、2 paired repetitions、greedy decoding、128 generated-token/tool-turn、8,192-token context、resource limits、停止条件を推論前に固定した。v1プロトコルは推論前レビューで起動時のsource/model/input照合不足を発見したため、**一度も推論せず**無効化し、記録を保存した。実行したのは照合を追加したv2のみ。
- v2 SHA-256: `ff51066704a0e4f31b05011d876722a73abab12654040b948e947930f23485fb`。
- モデルsnapshot: `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`, snapshot `019cc73c45c770444708a6dd8690c66243cc5c80`; `model.safetensors` SHA-256 `56a3d94706833f753e6c6b47ea57af8ef638cd8cc1d74eca1142ba640b26060e`。
- Backend/runtime: MLX Metal, mlx 0.32.4, mlx-lm 0.32.2, Python 3.12.12. tokenizer configはsnapshotのままで、template overrideなし。
- 合成fixture input SHA-256: `0dda649ee3829f8cb87a2158106a3a8876f62b25938e30a779cc5e712cd0e75b`。expanded prompt SHA-256: `36246781cf0639b165054b2513a018f53eda6e2efe301cf933c0b4318281cef1`。
- Offline tokenizer preflightで6段すべてのshared-prefix token IDsがstage promptの完全なprefixであることを確認した。turnごとの再利用prefixは2,980/3,092 tokens、最大full promptは3,191/8,192 tokens。
- 実行guardはprotocol、入力、スクリプト、model manifest、重み全体、clean worktree revision、両armのcontroller source hashを起動時に再照合した。raw generation prompts、model outputs、tool results、resource samplesを`run-002/`に保存した。

## 比較結果

| Pair | Arm | Attempt wall time | Full stage prompts | Amortized prefill tokens | Measured prefill incl. prefix build |
|---|---|---:|---:|---:|---:|
| 1 | v9 | 67.977 s | 18,669 | 18,669 | 66.967 s |
| 1 | v10 | 27.868 s | 18,669 | 6,637 | 26.772 s |
| 2 | v9 | 71.943 s | 18,669 | 18,669 | 70.964 s |
| 2 | v10 | 27.838 s | 18,669 | 6,637 | 26.709 s |

- 2/2 paired comparisons: raw action outputs and generated token IDs identical; both arms completed in every attempt.
- Pairwise wall-time reductions: 59.004%, 61.306%; mean descriptive reduction 60.155%.
- Pairwise prefill-token reductions: 64.449% in both pairs.
- Pairwise measured prefill-time reductions: 60.022%, 62.363%; mean descriptive reduction 61.192%.
- Model load: 0.877 s once for the full run. Actual lease-bound `read_lines` execution was approximately 0.00016–0.00029 s per attempt.
- v10 cache restore was below 0.001 s per attempt; shared-prefix cache build was 23.708–23.710 s across both turns. This build cost is included in the v10 prefill totals.
- v9 peak MLX Metal allocation was about 5.28 GB; v10 was about 6.22 GB. v10 trades higher transient Metal allocation for lower repeated prefill.
- 2 repeats on one synthetic fixture are not independent task families. No confidence interval or inferential significance claim is appropriate.

## Resource and stop-gate outcome

The run completed in 200.22 s with no abort. Peak process-tree RSS was 4,734,435,328 bytes (4.41 GiB); maximum swap growth was 0; minimum host-available memory was 7,998,767,104 bytes (7.45 GiB); minimum free swap remained 1,571,291,136 bytes (1.46 GiB). All remained within the unchanged limits: 10 GiB process RSS, 32 MiB swap growth, 2 GiB available-memory floor, 512 MiB free-swap floor, 16 GiB disk floor, 600 s/attempt and 750 s/run.

The preregistered local performance-use gate passed for v10: exact actions and token IDs, all interactions complete, exact prefix reuse without fallback, >20% prefill-token reduction in both pairs, no wall-time regression, and no resource abort. This GO applies only to this controller optimization on the measured inference shape.

## Coding-agent evidence boundary

This v9/v10 performance campaign did **not** search, edit, run public tests, or use an independent repair grader. It produced zero repairs.

A separate, previously frozen Sprint X synthetic feasibility run (`function_edit_e0_v5`, task `sprint-x-synthetic-function-e0-005`) did produce one model-generated function edit: 4/4 schema-valid turns, transactional edit applied, hash-bound source identity retained, and a restricted independent AST grader returned PASS. Its baseline failed, known fix passed, and wrong mutant failed. The grader never imported or executed candidate code. It is one synthetic family, not an external Development repair or held-out success.

No fresh eligible external Development family was available. Transformers #48319 was explicitly excluded before model inference because its existing qualification places it in the reserved held-out split; its source was not exposed to the model. Sprint IX MLX-LM #1773/#1777 remained consumed and were not reused. Thus new external repair families and independently graded external repairs in this sprint: **0**.

## Engineering and GitHub state

- PR #139 updated to `18afc60910ceb3ef5572fb2c7eadeff551bf9510`; exact-head CI passed for Python 3.11/3.12/3.13, lint, typecheck, blob budget, seeded summary, calibrated CPU tasks, RVL CPU inference, and actual optimizer/recovery CPU evidence. It remains Draft and stacked on #137; none of this code is in `main`.
- Targeted local schema-slot tests passed (99 tests); Ruff and `git diff --check` passed. The earlier full local Python suite had four Colima/Docker boundary failures because the Docker VM could not mount pytest's `/private/var/folders/...` temporary paths. The exact-head GitHub CI passed these suites.
- Main remains `f3ab90a36e8724441b6b9555255f8f75a0a4d529`; no PR merged.
- At live refresh there were 18 open PRs and 8 Drafts. #140 (`34f972e…`) and #143 (`6266ec0…`) were main-based with all visible CI checks green but `REVIEW_REQUIRED`/`BLOCKED`. The sole listed repository collaborator is the owner, so an independent approval cannot be requested from an available collaborator. #140's known P2 archive-symlink review concern appears fixed in its head but remains unresolved for reviewer confirmation. #143 had no submitted review. No branch protection was bypassed.
- The raw campaign, report, and hash manifest are appended to the existing Draft evidence PR #142; this does not make the implementation mainline code or resolve the PR’s independent-review/source-split blockers.

## Decisions

| Area | Decision | Evidence |
|---|---|---|
| v10 shared-prefix inference | **GO for measured optimization only** | Exact outputs; 64.449% fewer amortized prefill tokens; 59–61% lower attempt wall time; no resource abort |
| Synthetic edit path | **PARTIAL / positive feasibility** | One independently graded synthetic repair, separate from the v9/v10 run |
| Fresh external Coding Agent evaluation | **NO-GO / not run** | Zero eligible unconsumed Development families; reserved task excluded |
| Coding model improvement / held-out improvement | **NO-GO** | No valid disjoint learning/evaluation split and no external repair campaign |
| SFT / GRPO / policy updates | **NO-GO / not run** | No eligible coding reward groups; updates: 0 |
| Model-coupled long-horizon recovery | **NOT RUN** | No crash/restart, checkpoint, or optimizer experiment in this run |
| RSI research automation | **NOT RUN** | No agent-authored follow-on experiment decision loop |
| Mainline integration | **BLOCKED** | #139 remains stacked Draft; #140/#143 await independent review; main unchanged |

**Next highest-value step:** obtain a qualified independent reviewer for the main-based integrity PRs, and qualify new, unexposed external Development families that do not overlap reserved held-out work. The next model campaign should use the now-validated synthetic edit/tool path and the lower-prefill controller, then measure independently graded external repairs rather than extend this synthetic timing fixture.

## Reproduction

From the evidence checkout with the pinned local venv and cached model snapshot, the recorded command is:

```sh
/tmp/vare-sprint-vii-mlx/bin/python results/sprint_x_20261028/schema_slots_kv_cache_v1/reproduction/run_guarded.py
```

The guard expects the code worktree at `/tmp/vare-sprint-x-kv-cache-v1` on the pinned commit. The model weights are not included in Git; hashes and snapshot provenance are in `model-snapshot-manifest.json`. Run outputs are append-only and the reproduction script refuses to overwrite an existing run.
