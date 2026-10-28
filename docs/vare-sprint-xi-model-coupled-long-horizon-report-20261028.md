# VARE Sprint XI — 実モデル連結 Long-Horizon Systems Report

実施日: 2026-10-28
主な記録: `results/sprint_xi_20261028/`
研究コード: VARE PR #139, `df0a360ead3927f72d33972634251469b50c2368`
main: 開始・終了ともに `f3ab90a36e8724441b6b9555255f8f75a0a4d529`

## 結論

実モデル推論、実際のGRPO optimizer update、永続trajectory、実プロセスのSIGKILL、独立プロセスでのcheckpoint復旧を一つの小規模なSystems実験として接続した。独立監査は30/30項目を通過した。復旧したv2 checkpointは、faultなしの同期参照checkpointとadapter、Adam state、CPU RNG、checkpoint SHAが一致した。

これは合成報酬による一台・一seedのSystems結果である。Coding修復能力、外部タスク成功、held-out汎化、Coding RLの改善は測っていない。コントローラの広いagent軌跡に対する性能やasync throughputの優位も未測定である。

## 実モデルと固定条件

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- `model.safetensors` SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。tokenizer/config等の各ファイルhashはロック済みprotocolに記録した。
- Backend/runtime: Apple M1 Pro、CPU fp32、Python 3.12.12、PyTorch 2.9.1、Transformers 4.57.3、torch threads 1、offline、無償ローカル実行。
- Qwen base weightsは凍結し、学習対象はvocabulary-logit-bias policy adapterのみ。したがってこれは小規模な実ポリシー更新であり、基盤モデル全体のfine-tuningではない。
- 実験コードは固定revisionのRecursive-Verification-Lag (RVL) GRPO trainerとVARE replay、lag/freshness admission、checkpoint境界を使用した。PR未統合コードをmainline実装とみなしていない。
- 親protocol `vare-sprint-xi-qwen025-model-coupled-recovery-v1`と、停止後の未消費cohortだけを使うprospective continuation `vare-sprint-xi-qwen025-model-coupled-recovery-continuation-v1`を別々に保持した。Continuation protocol SHA-256: `ce783a3837792c0504110667eab5b35681ce59c3f7b91d5d28b82393db491caf`。Runner SHA-256: `db704017142a39ed0c3fb57fc5b3e7d9e926a9f0904dd3289b5b711c619fa7ec`。

## 実行経路と結果

経路は `Qwen rollout → durable artifact/manifest → frozen exact-choice verifier → VARE group-aware replay/admission → RVL GRPO step → adapter/Adam/RNG checkpoint → independent reload/evaluation → promotionまたはrollback`。

1. **Rollout worker recovery:** Qwen workerはfresh_v0 rolloutを生成・fsyncした後、acknowledgement前にSIGKILLされた。別プロセスで再開し、artifact hashを検証して既存trajectoryを再利用した。消費済みpromptを再生成していない。
2. **v1 update:** 実Qwen生成から非定数reward groupを得て、VARE replayを経由したGRPO optimizer stepを1回commitし、checkpoint `e44cbec0882ba50dfd984f12b83cf476f8c1ff437a9dc090c71582585987e5eb`を作成した。
3. **保存した初回ハーネス失敗:** queued stale cohortの8 promptはすべてreward 0で、ハーネスはfreshness判定前に「mixed groupがある」という誤った前提で`StopIteration`となった。更新済みv1、worker SIGKILL、失敗記録は保持した。欠けた結果を補うために消費済みpromptは再実行していない。
4. **Fresh v1 continuation:** 未使用だった事前宣言cohortを継続し、最初の非定数group `xi-v1-fresh-v1-03`を使用した。同期参照はv2 updateを1回commitした。
5. **Learner recovery:** 別learnerプロセスを実optimizer適用後、journalの`optimizer_applied_uncommitted` fsync直後にSIGKILLした。commit済みcheckpointは存在しなかった。v1 checkpoint、optimizer/Adam state、RNG、保存済みの同一rolloutを再ロードし、独立プロセスでupdateを再計算してv2を1回commitした。optimizer apply試行は2回だが、有効commitは1回。

### 完全性と定量値

| 指標 | 観測 |
|---|---:|
| 正常なVARE policy commit | v1、復旧v2の計2回 |
| v2同期参照 | 比較用の別branchで1 commit |
| learner fault | 実SIGKILL 1回、return code `-9` |
| worker fault | durable write後・ack前の実SIGKILL 1回 |
| recovery commit latency | 5.702秒 |
| recovered/reference checkpoint SHA-256 | 両方 `81e3efdd8aa58029e49a1e3c4657c2be1929c9c6b625220077d69f1e750e7dc2` |
| adapter identity | 同一: `9859304feae6281da33d10e9a75838de13fe594486013d52eec55605f8a582e3` |
| logical Adam state identity | 同一: `a10147da7fb3fbd6bb9d0377fd96b78341d4ed361062314f347e33d8fccbb752` |
| CPU RNG identity | 同一: `de7062258ababbe630dbf110be30165f669e235bfbb2b8cdb4c4cf3822adf3c5` |
| stale group freshness | VARE 0/4 admitted、naive admission-only 4/4 accepted |
| continuation campaign wall time | 44.27秒 |
| host swap growth | 0 bytes |
| post-run available RAM | 15,558,082,560 bytes |
| independent audit | 30/30 PASS |

stale groupは全報酬0だったため、どちらのarmもこのgroupでoptimizerを実行していない。naive 4/4は**admission-only対照**であり、naive async optimizer throughputまたは学習性能の比較ではない。

## 評価、promotion、rollback

固定4問のsynthetic arithmetic選択で、平均正解choice確率はv1 `0.0650586281`、同期v2と復旧v2は双方 `0.0661174757`。v2との差は約`+0.00105885`で、復旧と同期出力の数値的一致を確認した。この4問評価を一般能力・Coding性能・独立汎化の証拠とは扱わない。

事前に定めたnondecreasing synthetic choice score promotionを適用した。負の対照`-20` logit biasはscore `1.4341e-10`となって拒否され、active checkpoint identityは維持された。

## v10 shared-prefix KV cache

PR #139に8/16-turnの決定論的回帰fixtureを追加した。実tool schemaの`list_files`, `search_source`, `read_lines`, `replace_lines`, `check_syntax`, `run_public_tests`, `finish`を順に含み、変化するtool response、transient search errorとretry、prompt長変化、exact prefix clone、tokenization boundary mismatch時のcache fallbackを検査する。v9/v10のtool actionとstage token IDsは一致し、生成位置のprompt offsetも検証する。boundary mismatchではv10がshared prefixを使わずfull promptへfallbackする。関連テストファイル全52 tests、Ruff、diff whitespace checkはローカルで通過した。

この8/16-turn回帰は**決定論的mock fixture**であり、実モデルの8/16-turn生成やMetal/RAM長期測定ではない。別途Sprint Xで完了したQwen2.5-Coder-7B-Instruct-4bit / MLX Metalの2-turn比較は4 attempts・2 ABBA pairsでv9/v10の出力token identityが2/2 pair、prefill 18,669→6,637 tokens、wall time 59.0%/61.3%減、peak Metal allocation約5.28/6.22GB、peak RSS 4.41GiB、swap growth 0だった。Sprint XIはその過去結果を再分類せず、8/16ターンへ一般化もしない。

## Coding RL、RSI、held-out

- 新規適格Development family: 0。過去に消費したMLX-LM familyを再利用せず、reserved Transformers #48319にも触れていない。
- Sprint XIの外部repo agent attempt: 0。独立に評価された外部修復: 0。
- Coding rewardに基づくSFT/GRPO: NO-GO。実行した2 updateはsynthetic arithmetic rewardに限る。
- blind held-out improvement: NO-GO。歴史的な15-family eligibility gateも緩めていない。
- RSIのagent-driven仮説選択や適応的schedule: NOT RUN。

## GitHubと統合状態

- mainは開始・終了ともに`f3ab90a36e8724441b6b9555255f8f75a0a4d529`。Sprint XIでmergeは0件。
- PR #139 (`https://github.com/mitukx/VARE/pull/139`) にv10長軌跡fixtureをcommit `df0a360ead3927f72d33972634251469b50c2368`としてpushした。PRはDraft、baseは#137 head `6d95bd9196645ec7ecf7ff5d0c9ec997b24dcab1`。exact-head CIは20/20 success。独立reviewはなく、mainへは未統合。
- PR #142 (`https://github.com/mitukx/VARE/pull/142`) へ本レポートとSprint XI raw artifactsをcommit `f990817afeadfb8d331995a43e275625205f2bdd`で追記した。push時のGitHub blob-budget checkは3個のoptimizer checkpointが1 MiB超過で失敗した。この失敗記録は残した。続く修正で3原本をそれぞれ900,000 byte以下の2 partに可逆分割し、part連結後のsizeとSHA-256をcheckpoint manifestと照合する検証器を追加した。検証器は3/3原本の復元照合にPASSし、PRの現在差分から大blob原本を削除した。現時点の分割版はmanifest v2に記録済み。修正commitのexact-head CIはpush後に確認する。PRはDraftで、独立review/承認の代替ではない。
- main protectionを回避していない。#140/#143等を含むmainline integrationは適格な独立approval待ちであり、実行済みmainline成果はない。

2026-10-28のlive PR inventoryは開始時・終了時ともに18 open / 9 Draft。以下のmetadata snapshotと台帳は#142へのSprint XI追記前に取得した。CI列は取得時のGitHub status check数である。`clean`やgreen CIはレビュー承認や統合適格性を意味しない。

| PR | head → base | CI | disposition / 次の判断 |
|---|---|---:|---|
| #143 | `6266ec0a10` → main | 19/19 pass | MERGE-CANDIDATE。main-based、独立approval待ち。diff/export境界のレビュー後に統合 |
| #142 | `5fa0d8c553` → main | 19/19 pass | RESEARCH-EVIDENCE。既存証拠PRへSprint XIを追記。Draft/approval待ち |
| #140 | `34f972eb2a` → main | 19/19 pass | MERGE-CANDIDATE。空manifest対応。symlink/archive review指摘は実装済みに見えるがreviewer確認待ち |
| #139 | `df0a360ead` → #137 | 7 pass / 13 in progress* | BLOCKED。v10回帰commitを追加、Draft、#137への依存とexact-head CI完了待ち |
| #137 | `6d95bd9196` → #136 | 28/28 pass | MERGE-CANDIDATE。#136の後段として順序維持、#139 parent |
| #136 | `eaf3fd58dd` → #133 | 19/19 pass | MERGE-CANDIDATE。#133の後段として順序維持 |
| #135 | `b258c69fc0` → `codex/transformers-47468-qualification-v1` | 19/19 pass | BLOCKED。Draft。base lineageと親PRの現状を確定してから評価 |
| #133 | `e8ab4705b8` → `codex/mlx-pr4748-qwen3-pilot-v1` | 20/20 pass | REPAIR-FIRST。両P2の修正と回帰はheadにあり、関連37 testsも記録済み。独立reviewが未解決のため確認後にstackを整理 |
| #131 | `139de53819` → `codex/mlx-core-development-agent-feasibility-v1` | 20/20 pass | RESEARCH-EVIDENCE。Draft、pinned recovery/source inventory記録を保持 |
| #126 | `c728b9f2ec` → `codex/real-agent-interface-v1` | 20/20 pass | RESEARCH-EVIDENCE。verifier-aware async GRPOの履歴証拠を保持し、親に重ねて統合 |
| #122 | `37b3151981` → main | 28/28 pass | MERGE-CANDIDATE。main-based、独立approval待ち |
| #121 | `a937f4ad6e` → main | 19/19 pass | MERGE-CANDIDATE。main-based、独立approval待ち |
| #120 | `8aacae54d0` → main | 20/20 pass | REPAIR-FIRST。evaluator completion spoof P1にはscope-bound regression/controlがheadにあり、独立確認待ち。empty inventory P1はfocused #140へ抽出済み |
| #118 | `552d6f6118` → `codex/transformers-49752-repo-task` | 20/20 pass | EXTRACT。structured edit libraryを独立統合するには親source/task branchとの依存解消が必要 |
| #117 | `6709aed996` → #114 | 19/19 pass | RESEARCH-EVIDENCE。Draft、snapshot isolation/exportのstacked lineageを保持 |
| #114 | `8257a4dfe2` → #99 | 19/19 pass | RESEARCH-EVIDENCE。Draft、calibration publication failure recordを#99 ancestryとともに保持 |
| #108 | `710dbc0d28` → `codex/fresh-hf-dev-candidates-v1` | 19/19 pass | RESEARCH-EVIDENCE。Draft、dual-Python protocol lineage。freshness/exposure gateの再確認後に続行 |
| #99 | `762a5a6559` → main | 20/20 pass | RESEARCH-EVIDENCE。Draft qualification historyを保護し、exposure/reserved-split gateなしにDevelopment taskとして使わない |

\* #139 status snapshot時に2 workflow runsのchecksが同時進行中だった。push後に再度exact-head CIを確認した。

main protectionは1件のrequired approvalを要求し、`enforce_admins=true`。利用可能な適格独立reviewerが確認できなかったため#140/#143/#122/#121等の承認は依頼・偽装せず、mergeもしていない。Sprint XIでは#139と#142のみ更新し、open/draft数は変えていない。既知のP1/P2を含むPRもあるため、Sprint XI内で「green CIだから統合」とは判断していない。

全18件のhead/base SHA、Draft/review/merge状態、check counts、disposition、次の判断を含む機械可読スナップショットは`results/sprint_xi_20261028/github-open-pr-ledger-20261028.json`に保存した。

## 最終判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| v10 8/16-turn controller correctness | PARTIAL / local GO | 決定論的regressionは通過。実モデル長軌跡・長期memoryは未測定 |
| 実モデルrolloutとdurable worker recovery | RUN / PASS | Qwen rolloutをworker SIGKILL後にartifact再利用 |
| 実optimizer updateとlearner recovery | RUN / PASS | GRPO adapter update、実learner SIGKILL、recovered/sync state完全一致、30/30監査 |
| verifier freshness | RUN / PARTIAL | stale 0-reward groupでVARE 0/4対naive 4/4 admission-only |
| Long-Horizon RL capability | NO-GO | one-host/two effective updates/synthetic reward。長いcoding trajectoryやpolicy capabilityは評価なし |
| 外部 Coding Agent repair | NO-GO / NOT RUN | eligible fresh Development familyが0 |
| Coding SFT/GRPO・held-out improvement | NO-GO / NOT RUN | coding rewardと適格なsplitなし |
| RSI research automation | NOT RUN | agentによる次実験選択なし |
| mainline integration | BLOCKED | main保護と独立approval不足。merge 0 |

## Reproduction pointers

- Parent protocol/run/failure: `results/sprint_xi_20261028/model_coupled_qwen025_v1/`
- Prospective continuation protocol, independent worker/learner logs, checkpoint, audit and manifest: `results/sprint_xi_20261028/model_coupled_qwen025_recovery_v1/`
- Sprint-wide SHA-256 inventory v1 (`results/sprint_xi_20261028/manifest-sprint-xi-v1.json`) is retained with its `.sha256` sidecar in the original evidence commit `f990817afeadfb8d331995a43e275625205f2bdd`. That version captured the three raw 1,218,465-byte optimizer files before the repository blob-budget check. The follow-up stores those same bytes as six 900,000-byte-or-smaller parts; `checkpoint-optimizer-parts-v1.json` and `verify_checkpoint_parts_v1.py` reconstruct and verify each original SHA-256 against the checkpoint manifests. Follow-up Sprint-wide manifest v2 inventories the current split representation and updated report.
- Reproduction runners: `scripts/sprint_xi/run_model_coupled_v1.py`, `scripts/sprint_xi/run_recovery_continuation_v1.py`, `scripts/sprint_xi/audit_recovery_continuation_v1.py`.
- Verify the lossless optimizer-file split with `python scripts/sprint_xi/verify_checkpoint_parts_v1.py`; pass a new temporary `--materialize-root` to reconstruct the original `.pt` bytes outside the evidence checkout.
- Continuation command: `python scripts/sprint_xi/run_recovery_continuation_v1.py --protocol results/sprint_xi_20261028/model_coupled_qwen025_recovery_v1/protocol.lock.json --original-protocol results/sprint_xi_20261028/model_coupled_qwen025_v1/protocol.lock.json --base-run results/sprint_xi_20261028/model_coupled_qwen025_v1/runs/run-001 --output <new-output>`.
