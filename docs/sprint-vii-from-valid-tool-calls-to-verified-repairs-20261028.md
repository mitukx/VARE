# VARE Sprint VII — 実験・根本原因・統合結果

**日付:** 2026-10-28 (JST)

**リポジトリ:** [mitukx/VARE](https://github.com/mitukx/VARE)

**研究記録:** `results/sprint_vii_20261028/`

**主目的:** 有効なtool callから、実source edit、独立検証まで到達する最小ループの実証

## 要約

今回、M1 Pro上のQwen2.5-Coder-7Bで二種類の合成fixture用編集APIを実行し、両方でモデル生成のhash-bound editを適用し、別プロセスのAST graderでPASSを得た。これはagent実行経路を検証する**合成fixtureのcontroller evidence**であり、一般的なCoding Agent能力の証拠ではない。両attemptは編集後に`finish`せず、8ターンを使い切った。

次に、新規のTRL Development task familyをbaseline/fix/wrong-mutantで認定してから、同じ7B checkpointで1 attemptだけ実行した。14/14 tool actionはschema-validだったが、モデルは有用なsource位置を見つけられず、同じ検索を繰り返して編集せずに停止した。候補はbaselineとbyte-identicalであり、public testとcandidate graderは実行していない。**独立に認定された新規外部修復は0件。** 指示した停止規則どおり、別タスクを消費せず、記録と診断に戻った。

E0 deterministic edit-acceptance v3は16項目すべてを通過した。探索失敗を受けたtransactional navigation guidance v2を実装し、関連回帰188件、Ruff、構文検査を通した。v2は実モデルで未検証である。更新コードはDraft PR #139へpushしたが、mainは変更していない。

## 1. Sprint VIの再監査とroot cause

歴史的なSprint VI記録は変更していない。詳細なattempt単位のmatrixは[根本原因matrix](../results/sprint_vii_20261028/root-cause-matrix-v1.json)に保存した。

| 過去attempt | 実際に止まった段階 | 根拠と原因 |
|---|---|---|
| Transformers #49696 / Qwen2.5-3B autonomous | argument generation | 512 tokenでargument slotが完了せず1 turnで停止。tool callもsource readもなし。 |
| #49696 / source-read-gate | query validation | 239 token後、引用符付きの複数行regex風queryが`invalid_query`。JSON構文はtoolが受け取れるqueryを保証しない。 |
| #49975 / 3B pair | model/tool-controller boundary | 512 tokenずつで停止したとのfrozen assessment。公開bundleには`agent.json`がなく、parserとmodelのどちらが原因かを再構成できない。これ以上細かい断定はしない。 |
| #50001 / 7B autonomous | source localization/context | 11 calls中、同一`import BasicTokenizer`検索を繰り返し、先頭1–100行を4回読んだ。92生成token、編集なし、`context_budget_exhausted`。 |
| #50001 / localized | argument generation | 528 token、read 1回の後にargument slot上限で停止。編集なし。 |
| #49900 / 14B | source-to-edit transition | search/read後、541 tokenでargument slot停止。編集なし。別にwrapperがempty Git stdoutから誤った非empty diffを作った。append-only reconciliationでraw sourceから再計算し、誤diffとgradeを除外。 |
| #49842 / 14B（過去露出のためfresh結果から除外） | edit validation | 12 schema-valid turns。`normalization=none`のままPython code fenceを含むeditを4回出し、構文validatorがすべて拒否。paired localized armはswap増加653.5 MiBで、固定512 MiB上限により中断。 |
| Sprint VII external / 7B | source localization/action selection | 14 schema-valid calls。最初の検索はno-hit、その後同じ検索を10回出して`duplicate_query`。編集なし。 |

この再解析から、失敗段階を一括して「モデルがコードを書けない」とは扱えない。3Bの一部はgeneration contract、#49696はquery validation、#50001と今回の外部attemptはnavigation/action selection、#49842はedit normalization、#49900は別個のartifact-wrapper correctness問題だった。#49975についてはraw record不足を限界として明記する。

## 2. Deterministic edit-acceptance E0

`vare.sprint-vii.e0-edit-acceptance.v3`を実行し、`e0_run_03/e0-result.json`に結果を保存した。結果は**PASS 16/16**。

- hash-bound source read → edit → source hash change → canonical diff → independent workspace verificationが成功。
- baselineはFAIL、known fixはPASS、wrong mutantはFAIL。
- invalid patch、stale source、symlink、grader spoof、grader/source driftを拒否。
- graderはcandidateをimport/実行せず、固定AST形状を解析。
- graderは実験前後で同一hash。モデル推論は行っていない。

これはtrusted edit/evaluator経路のdeterministic evidenceであり、モデルの修復成功数には含めない。

## 3. Model–tool compatibility: 合成fixture

**モデル:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`

**snapshot:** `019cc73c45c770444708a6dd8690c66243cc5c80`

**backend:** MLX Metal、Python 3.12.12、MLX 0.32.4、mlx-lm 0.32.2。モデル実行journalではMLX version欄が`unknown`だったため、package versionは事前preflight記録による。

**tokenizer/template:** snapshot内の設定を使用し、overrideなし。effective chat-template SHA-256 `d5495a1e5db0611132a97e46a65dbb64a642a499421228b9c8b93229097fa9a4`、元の`tokenizer_config.json` SHA-256 `f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7`。prompt本文と全raw outputはevent journalに保存。

**frozen protocol:** `model_interface_v3.json` SHA-256 `27796d62cd20a5764ba5191804df499c8a161b61d87afe55ad46378442490303`。

同じ合成task `shift(value) -> value + 2`で、string-body v1とarray-body v2を比較した。v1は51 generated tokens / 8 turns / 111.478s / 4.56 GB peak RSS、v2は53 tokens / 8 turns / 114.411s / 4.77 GB peak RSS。どちらも有効なschema action、hash-bound edit適用、byte-exact source change、canonical diff再構成、独立AST grader PASSに到達した。swap growthは両方0。

ただし両armとも編集後に`list_functions`を繰り返して8-turn limitを使い切り、`finish`しなかった。public testは公開せず、実taskではない。従ってこれはAPI compatibilityとedit acceptanceの成立を示すが、外部コード修復、一般能力、post-edit completionを示さない。

postprocessorが空のGit diffを落とす既存bugもこのrunで再現した。歴史的agent outputは変更せず、別auditが固定source SHAから176-byte canonical diff（SHA-256 `245693fd9b9e222691fcbd5a39fda88d238b86d7987e35a1eca428072a025225`）を再構成し、両synthetic candidateを独立gradeした。wrapper修正では`--no-ext-diff --no-textconv`を指定し、raw Git bytesからdiffを取る。

## 4. 新規TRL Development taskと実モデルattempt

**Upstream task:** [TRL PR #7913](https://github.com/huggingface/trl/pull/7913), CPO/ORPOでcompletionがすべてtruncatedされたtraining exampleを除外する修正。1 source-fix familyとして数えた。

**baseline SHA:** `1374d6956419455864dd642911c7d4a0334e60b0`

**修正PR head:** `3ed1ae5ac17076ff67d4be24cbd303c528081b4d`

**merge SHA:** `33b871b5eee4d7a51dcc42c1b5454a4d0364339a`

**candidate source:** `trl/experimental/cpo/cpo_trainer.py`, SHA-256 `41a52900b1761949b02d3bb13975fcbad9a69edb4b7f5fc23eec7b6c39d3568d`。

同一upstream testでbaseline FAIL / known fix PASS / wrong mutant FAILを確認した。隔離container内のstrict-scope graderはknown fixをPASSし、wrong mutantをcandidate実行前にrejectした。candidateにはeditable sourceだけを配置し、grader・known fix・hidden testsは露出していない。既存recordに検索/露出があったMLX-LM #2014はmodel exposure前に除外した。TRL taskはDevelopmentであり今回のattempt後は消費済み。pretraining exposureはUNKNOWN。held-outには使わない。

**固定protocol:** `protocol.v2.json`, SHA-256 `3c4ac0ee5040abb88b18f2262ee248a8165eb8af75210e64ab858a9b9766ef86`。1 attempt、seed 20261028、greedy、14 turns、192生成token/turn、1536 total生成token、8192 context、480s agent limit、RSS 10 GiB上限、swap growth 32 MiB上限。preflight時に利用可能memory 9.65 GB、free swap 1.41 GB、disk free 45.76 GB。上限は変更していない。

再現性上の制約: attempt recordにはcontroller source hashがあるが、生成直前の`run_local_agent.py` bytes自体はattempt archiveへ同梱されていない。record上のhash `bb0894ec…`はPR #139の最初のinterface commit `ec6f37ce`に保存されたrunner (`f538d5a4…`) と一致しない。raw trajectoryと結果は保持されているものの、当時のrunner全体をcommit SHAだけで完全再構成できるとは主張しない。

**結果:** Qwen2.5-Coder-7Bの同一snapshotで14/14 actionがschema-valid。1 `list_files`、2 `read_lines`、11 `search_source`。最初は引用符付き複数語query `"dataset preparation CPOTrainer"`を出しhitなし。その後10回同一queryを繰り返し、すべて`duplicate_query`。最初の1–100行を2回読んだが、検索hit付近のreadに進まなかった。generated tokens 144。各constrained decoding stageの最大promptは4,919 tokensで8,192上限未満。全stage prompt tokensの合計98,665は、反復したgeneration stageの処理量でありcontext window長ではない。

source edit attempt 0、applied edit 0、candidate diff 0 bytes、candidate sourceはbaselineと同一。public testは公開しておらず未実行。candidate graderも`NOT_RUN_NO_CANDIDATE_PATCH`。wall time 362.222s、peak process-tree RSS 4,381,933,568 bytes、minimum sampled available RAM 6,296,272,896 bytes、swap growth 0。resource abortなし。従って主因はresourceでもJSON/schemaでもなく、source localizationとrepeated action selection。指示されたstopping ruleに従い、同一taskへのprompt retryや追加task消費を行わなかった。

## 5. 仮説に基づくコントローラ変更

v1 transactional guidanceは「sourceをsearchし、narrow rangeを読む」としか指示せず、queryがline-orientedであること、検索失敗時の短縮、重複query停止を説明していなかった。`scripts/run_local_agent.py`に明示指定できるnavigation guidance v2を加えた。v2は短い単一identifier、同一line一致、no-hit時の短縮再試行、duplicate query禁止、hit付近のnarrow readを指示する。既存default v1は維持し、edit tool schemaも変えていない。transactional one-time lease、explicit normalization、全file syntax validation、fail-closed write semanticsは維持する。

Regression: guidance/navigation関連188 tests passed、Ruff・py_compile・`git diff --check` passed。#139のexact-head CIで追加のfixture欠落が2件判明したため、fixtureを修正して`e1be20cbf67d4d60a3e180172e9598d4bc0d1862`をpushした。修正後のcontroller/navigation/agent grader関連suiteは58 passed、4 skipped。実モデルではguidance v2を**未実行**。次のreal-model testには、未露出のfresh Development familyと独立qualificationが必要。

## 6. GitHub/PR統合

| 対象 | live state / このSprintの判断 |
|---|---|
| `main` | 開始・終了SHA `f3ab90a36e8724441b6b9555255f8f75a0a4d529`。変更なし。 |
| [#139](https://github.com/mitukx/VARE/pull/139) | Draft、baseは#137系branch。`ec6f37ce`でagent interfaces、`e5f3be9c`でdev NumPy依存とmypy narrowing、`e1be20cb`で2件のtest-fixture不備を修正。現head `e1be20cbf67d4d60a3e180172e9598d4bc0d1862`の必須check 9件、calibrated CPU、actual optimizer/recoveryがすべてpass。独立approvalとstacked dependencyの解消待ち。 |
| [#133](https://github.com/mitukx/VARE/pull/133) | head `e8ab4705b84d82cd79e2883c391083087e82b4e8`、closed-unmerged #132のbranchにstacked。path inventory上限とtransient index retryの2件のP2 threadは未解決だが、該当修正はこのheadに存在し、全表示CI checksはpass。mainへ入っていない。 |
| [#137](https://github.com/mitukx/VARE/pull/137) | #136にstacked。前回監査時exact-head CI成功。approval/dependency解消なし。 |
| [#120](https://github.com/mitukx/VARE/pull/120) | head `8aacae54d0f8893580a77cb8bb831e7621a2403a`、全表示CI checks pass。completion forgery P1とempty-manifest P1 threadは未解決。manifest修正は#140に抽出済み。P1完了は独立レビュー未確認のためtrusted reward pathへ入れない。 |
| [#140](https://github.com/mitukx/VARE/pull/140) | head `34f972eb2a300711f36fd81a72b6948528dfebad`、全表示CI checks pass。symlink/archive fallbackとduplicate-key修正はheadに存在し、P2 threadは未解決のまま reviewer confirmation待ち。 |
| [#143](https://github.com/mitukx/VARE/pull/143) | head `6266ec0a10920ae9ef95ccb528145ef14a452e39`、全表示CI checks pass、open review threadなし。必須の独立approval待ち。 |
| [#142](https://github.com/mitukx/VARE/pull/142) | Draft evidence archive、head `d51deae3408bbdf1835eee0c5072cbe1ac836133`。Sprint VII report、protocols、curated raw trajectories、task/grader control receiptsを追加。現headのtest 3.11/3.12/3.13、lint、typecheck、grader、CPU checksはすべてpass。`REVIEW_REQUIRED`で独立承認待ち。 |

`ec6f37ce`のCIではPython matrixのtest collection時にNumPyが不足し、typecheckでも`schema_slot_protocol.py`の`Any | None` list typingが失敗した。dev extraへ`numpy>=1.26,<3`を追加し型絞りを明示した`e5f3be9c`では、3.12/3.13のtest jobsがそれぞれ2件失敗した（3.12: 1,828 passed/112 skipped、3.13: 1,818 passed/122 skipped）。原因は、controller identityテストfixtureがstartup-hashed moduleを作っていなかったことと、agent preflightの仮workspaceにcandidate diff用Git `HEAD`がなかったこと。後者は本番のfail-closed validationが意図どおり先に停止した結果だった。テストfixtureを補い、baseline commitはcandidate-visible sourceと一致するbytesから作るよう変更した。`e1be20cb`後のローカル関連suiteは58 passed/4 skipped。失敗記録は[CI failure record](../results/sprint_vii_20261028/github_ci/pr139_e5f3be9_failure_record.json)とPR #139のGitHub Actionsに保存した。修正後のexact-head `e1be20cb`では必須checks 9/9と`calibrated-cpu-tasks`、`actual-optimizer-and-recovery`がpassし、3言語版Python matrixは各2 runともpassした（run IDs `40622442697`, `40622449164`, `40622449205`, `40622449291`）。機械可読な結果は[exact-head CI summary](../results/sprint_vii_20261028/github_ci/pr139_e1be20cb_ci_summary.json)に保存した。PR #142の現head `d51deae`でも表示されている必須checksはpass。Ruffとmypy（86 source files）もpass。live PR数は18 open / 9 Draft。protected mainは1 independent approving reviewと必須checksを要求する。Sprint VIIでの実操作はPR #139を3 commit更新、PR #142に1 evidence commit追加、0 merge、0 close。approval、branch protection、review threadは迂回していない。

## 7. RL / Long-Horizon / RSIゲート

- **Coding-reward SFT/GRPO:** NO-GO / NOT RUN。新しいexternal verified repairがなく、eligible coding reward groupもない。
- **Held-out improvement:** NO-GO / NOT RUN。過去のreadinessは9/15で15-family gate未達。Development taskをheld-outに流用していない。
- **Model-coupled long-horizon recovery:** NOT RUN。合成agent fixtureの8-turn tracesにはdurable recovery cutpoint比較がなく、#139/#133のsynthetic continuation結果をcoding capabilityへ読み替えない。
- **RSI loop:** NOT RUN。failure taxonomyを記録しただけで、agent選択実験から独立評価済みの次の改善には到達していない。

## 8. 最終判定

| 出力 | 状態 | このSprintの証拠 |
|---|---|---|
| Deterministic edit acceptance | `RUN / PASS` | E0 v3、16/16 deterministic controls。 |
| Real-model schema/tool generation | `RUN / PARTIAL` | synthetic fixtureとTRL taskでschema-valid actions。 |
| Synthetic model-generated source edits | `RUN / PASS` | 2/2 API arms applied + AST grader pass。ただしtoy fixture限定、終了失敗あり。 |
| Fresh external Coding Agent task | `RUN` | TRL #7913 task familyを認定し、1 Qwen 7B attemptを実行。 |
| Fresh independently verified repair | `NO-GO` | 0 task families; external candidateなし、grader未実行。 |
| Task-independent navigation intervention | `RUN / PARTIAL` | guidance v2実装と188回帰PASS、real-model validationなし。 |
| SFT/GRPO / held-out improvement | `NOT_RUN / NO-GO` | 学習適格性・15-family gate未達。 |
| Long-horizon model/recovery comparison | `NOT_RUN` | model task completionまで到達せず、durable-vs-naive比較なし。 |
| RSI research automation | `NOT_RUN` | 実験選択による改善なし。 |
| PR closures / merges | `0 / 0` | main SHA unchanged。 |
| PR update | `RUN / PARTIAL` | #139に`ec6f37ce`、`e5f3be9c`、`e1be20cb`をpush。#142にevidence archiveを更新。両PRの独立レビューが残る。 |

**厳密に新しく示せること:** trusted hash-bound edit pathと独立graderは、実際のQwen2.5-Coder-7B生成でsynthetic source editを2回受理できる。外部taskではschema generationは成立したがnavigation段階で停滞し、独立修復には至らなかった。従って現在主張できるのは「モデル生成から安全な編集適用へのsynthetic feasibility」と「外部Development attemptでの具体的なnavigation失敗測定」であり、Coding RLの改善、Long-Horizon RLの実モデル復旧、RSIによる能力向上ではない。
