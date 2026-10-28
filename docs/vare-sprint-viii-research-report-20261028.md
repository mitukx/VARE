# VARE Sprint VIII — 進捗を持つCoding Agentの実験報告

実施日: 2026-10-28
対象ブランチ: `codex/development-navigation-campaign-v1`
対象PR: [#139](https://github.com/mitukx/VARE/pull/139)

## 要約

このSprintでは、実験前のruntime identity、タスク資格審査、編集後の終了動作、外部リポジトリ上の実モデル軌跡を調べた。Qwen2.5-Coder-7B-Instruct-4bitは外部タスクで構文制約を守る有効なtool callを10回生成したが、同じ失敗検索を7回繰り返し、source edit、syntax check、`finish`に到達しなかった。独立graderは変更されていないbaselineをFAILと判定した。これは有効なnegative resultであり、Coding RLの成功や修復能力の証拠ではない。

合成fixtureではモデルが正しい編集を3/3回作り、独立AST graderが3/3回PASSした。post-edit gateを含む2条件ではモデル自身が`finish`を呼び、現行controller条件は8 tool callsを使い切って終了しなかった。この差は1つの合成fixture上の診断に限られ、実リポジトリ修復とは区別する。

独立承認がなく、Sprint VIII中のmain mergeは0件。PR #139は`#133 → #136 → #137 → #139`のstacked chain上にあり、mainline統合ではない。

## 1. 開始時点とGitHub状態

- `origin/main`はSprint開始時・確認時ともに`f3ab90a36e8724441b6b9555255f8f75a0a4d529`。
- 開いているPRは18件、Draftは9件。Sprint VIII中にclose/mergeは行っていない。
- PR #139はDraft、base `codex/navigation-integrity-fixes-v1`、headはコード修正後の`cac34bc5ec2f5580582e5a49825ae3b4a6e5147f`。レビュー承認なし。
- このコード修正headでは必須CI contexts（Python 3.11/3.12/3.13、lint、typecheck、blob budget、RVL inference、seeded summary、Math500 grader）がすべてpassした。実験記録と報告を追加した新headではCIが再実行される。
- mainの保護には1件の承認と9個のstatus contextが設定されている。GitHub collaboratorとして確認できたのはrepository owner `mitukx`のみで、独立reviewerはこの実行環境から依頼できなかった。保護設定は変更せず、自己承認も行っていない。

| PR | Sprint VIII監査時のhead | 状態・阻害事項 |
|---|---|---|
| [#120](https://github.com/mitukx/VARE/pull/120) | `8aacae54d0f8893580a77cb8bb831e7621a2403a` | main-based。自動reviewとownerのCOMMENTED記録はapprovalではない。completion forgeryとempty manifestのP1指摘に独立reviewが必要。 |
| [#133](https://github.com/mitukx/VARE/pull/133) | `e8ab4705b84d82cd79e2883c391083087e82b4e8` | baseは閉鎖済み#132系統。index output overflowとtransient index failureのP2 threadが未解決。後続stacked PRの基点。 |
| [#136](https://github.com/mitukx/VARE/pull/136) | `eaf3fd58ddd6fdf67f6d17d2f25a67955784a84f` | #133上のstacked変更。 |
| [#137](https://github.com/mitukx/VARE/pull/137) | `6d95bd9196645ec7ecf7ff5d0c9ec997b24dcab1` | #136上のstacked変更。 |
| [#139](https://github.com/mitukx/VARE/pull/139) | `cac34bc5ec2f5580582e5a49825ae3b4a6e5147f` | Draft、#137上。CIは一部実行中の時点で確認。 |
| [#140](https://github.com/mitukx/VARE/pull/140) | `34f972eb2a300711f36fd81a72b6948528dfebad` | main-based。checksはgreenだがarchive fallback symlinkに関するP2 threadと独立approvalが残る。 |
| [#142](https://github.com/mitukx/VARE/pull/142) | `0ac1857d934dcfe9e463761234d1da924acbcba1` | main-based Draft。CIはgreen、approval待ち。 |
| [#143](https://github.com/mitukx/VARE/pull/143) | `6266ec0a10920ae9ef95ccb528145ef14a452e39` | main-based。checksはgreen、独立approval待ち。 |

上記以外を含む全PRはAPIで件数とheadを再確認した。独立承認が必要なmain-based PRをmerge可能とは扱っていない。PR #139のexact-head必須checksは報告追加後にも再確認する。

PR状態表はSprint VIIIの実験・証拠公開開始前の監査スナップショットである。この報告とraw evidenceの追加により#142のheadは更新される。公開後のheadとCIはGitHub上のPR記録を参照する。

## 2. 実装とruntime provenance

### 修正したtool guidanceの不整合

`scripts/run_local_agent.py`のtransactional edit guidanceは、実際に公開されたtool集合を見ずに`run_public_tests`を案内していた。公開テストtoolがないタスクでは、この指示は実行不能である。利用可能なtoolに応じて、テストtoolがあるときだけテスト実行を案内し、ないときは`check_syntax`後に独立graderへ提出するよう変更した。

回帰テスト`test_transactional_guidance_matches_available_verification_tool`を追加し、tool集合とguidanceの対応を確認した。作業commitは`cac34bc5ec2f5580582e5a49825ae3b4a6e5147f`。ローカルの関連回帰群はpytest plugin自動読込を無効にしてpassした。通常のplugin自動読込では、global Hydra/OmegaConfとANTLRのversion不整合でtest collection前に失敗した。これはテスト対象の失敗ではなく、再現コマンドの環境条件として記録する。

証拠PRの最初のexact-head CIではPython 3.13の2 testが、standalone campaign runnerのoptional `psutil` importで失敗した。実験時にprotocol v5が記録した旧test source（SHA-256 `75314a676e33c8d26503a90fa05111a9f493f33e677879489a7699b12f9f52ad`）を`tests/fixtures/sprint_viii_campaign_exposure_v1.py`へbyte-identicalに保存し、現在のhelper testsではresource-monitor dependencyをstubしてimportするよう修正した。変更はtest loaderだけで、凍結済みcampaign runner、protocol、model outcomesは変更していない。修正後の2 testはlocalでpassし、最新PR CIを再実行中。

### 実験時の実装同一性

外部campaign v5はrunner/controller/graderとimportされたVARE moduleの実ファイルhashを凍結し、source snapshotを保存した。independent runtime auditは`PASS`、runtime manifest SHA-256は`38af64e499a174af715857c0a8924c063dd4c23098878a72d76e87f8fa01e026`。ただし実験時checkoutはdirtyで、auditにも`worktree_dirty=true`および`runtime_sources_dirty=true`が記録されている。実行bytesはmanifestとsnapshotに結び付いておりhash mismatchは報告されていないが、Git commit単独では実行状態を再構成できない。Sprint VIIの旧attemptに残るrunner bytes不完全性も修復・書換えしていない。

campaign v5が実際にロードした`run_local_agent.py`はSHA-256 `24f971d63306b9cdab56fddbb7ad8b5ed4ea1024cf0e604c885fb1bb93e1f179`。実行時のGit HEADは`fd98fa77e28874a19baa126b5e4d546697037a39`。上記guidance修正はその後のcommitに含まれるため、campaign v5のモデル軌跡を修正版runnerによる測定として扱わない。

## 3. 外部Developmentタスクと実モデルcampaign

### タスクと独立grader

- Repository: [huggingface/transformers](https://github.com/huggingface/transformers)
- Issue / upstream PR: [#49698](https://github.com/huggingface/transformers/issues/49698), [#49703](https://github.com/huggingface/transformers/pull/49703)
- Baseline revision: `8e40ad7bc7019ba19ae8b0e7d98e60376d0fac2a`
- Fix revision: `094669a133c54bafd2975c11a325140bc93b3459`; merge commit `f348d3dcfa9317bed603e01b8ca22b11fe59615c`
- 修正内容: Mask2Formerのprojection listを`nn.ModuleList()`で登録する。
- Task family: `transformers-49698-mask2former-module-registration`（1 family）
- Baseline source SHA-256: `6d69697d35325a576618b7e5f0d08aeb51705625ed0c22a753c1314a36df74bc`
- Known-fix source SHA-256: `929838b2bc993c420859aa0722f2f93738517f4609da61de479d618f6a3a8eb8`

candidate workspaceには対象の1 source fileだけを置き、upstream Git history、known fix、graderは含めなかった。独立AST graderはbaseline FAIL、known fix PASS、誤った`nn.ModuleDict` mutant FAIL。実際のTransformers classを使う校正では、baselineのprojection state parameter数0、修正後6で、3段identity branchを維持した。これはtask/grader資格審査の結果であり、agent修復成功ではない。

prior VARE artifact scanではtask family重複なし。upstream fixは公開済みのため、モデルのpretraining exposureは`UNKNOWN`。Development専用であり、blind held-outには割り当てない。

### 凍結条件とprotocol履歴

campaign v5 protocol SHA-256: `4295dddbbd70648ef003b6246aed4e502d057864463128591ee6ae69c43232a4`。Qwen2.5-Coder-7B-Instruct-4bitの同一snapshot、同一task family、greedy decoding、seed `20261028`、turn上限12、turn毎128 token、attempt総計768 token、context 8192、agent wall 240秒、outer attempt 300秒、process RSS上限10 GiB、swap増加上限32 MiB、free swap下限512 MiB、free RAM下限2 GiB、free disk下限16 GiBを固定した。model snapshotは`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`、revision `019cc73c45c770444708a6dd8690c66243cc5c80`、MLX Metal、4-bit。実行環境はPython 3.12.12、MLX 0.32.4、mlx-lm 0.32.2、Transformers 5.21.0、Torch 2.9.1、Apple M1 Pro / 32 GiB。

v1〜v4はmodel process開始前に検出したprotocol/runner不整合で無効化され、記録は残している。protocol SHAは順に`1e6c6940815dd9bd7937b5017dfc5f1e3ce2f030ba368ded5322884d35c4ff40`、`c7d218d855b302311fe32961917371cec3615e12ec81ec1772dc417721ae65aa`、`e8d387ae86c33c5eeb2100665c15fe4a12912a4164024ae71560035790ac1ad0`、`103320a4127db8dad9ac841b78758d1738758dbc295ebec0e408b85e1daccc40`。理由は公開toolとguidanceの不一致、instruction key mismatch、budget key mismatch、protocol ID mismatch。いずれもmodel inferenceなし、task exposureなし。v5は専用`--preflight-only`でmodel未起動のまま検査を通してから実行した。

### 実測結果

| arm | モデル実行 | tool trajectory | edit / grading | 終了状態 |
|---|---|---|---|---|
| Autonomous navigation | 実行。schema-valid call 10/10、generated 95 tokens、elapsed約251秒 | 初回search、`list_files`、先頭1–100行のread、その後同じsearchを7回再送。unique query 1、duplicate-query rejection 7、valid read 1 | edit attempt 0、applied edit 0、syntax/public tests 0。unchanged baselineを独立graderがFAIL | `finish`なし。wall-time exhaustion、controller finalizationなし |
| Source-localized | model未起動。runnerが1.18秒でexit 2 | `--navigation-guidance v2`がlocalized interfaceでは非対応 | candidate未変更。baselineをgraderがFAIL | runner failure。model outputなし |

Autonomous armのindependent grader receiptは`projection_container_is_not_an_empty_module_list`でFAIL。public testsはタスクに設定されていなかったため、`check_syntax`も使われなかった。strict tool schemaは正しい引数形を保ったが、意味のある検索語への切替を保証しなかった。trajectoryには同一検索へのduplicate rejectionが返された後もモデルが同じ検索を繰り返す挙動が残っている。

source-localized armの失敗はmodel failureではなくcampaign driverのinterface mismatchである。protocol上このarmもexposure markerを記録し、同じfamilyはすでにautonomous armに提示済みなので再実行しない。このためmatched comparisonは不完全で、arm間の効果差を算出しない。

### 資源

Autonomous armでpeak process-tree RSSは4,641,210,368 bytes（約4.32 GiB）、MLX peak memoryは5,352,465,576 bytes（約4.98 GiB）、attempt中の最小host available memoryは約6.24 GB、swap増加0 bytes。campaign終了時のfree diskは約43.9 GB、free swapは約1.44 GB。resource abortは0。設定上限を緩和していない。

独立監査とSHA manifest: `results/sprint_viii_20261028/external_transformers_49703/audit/campaign-v5-independent-audit.json`（SHA-256 `5f6c7beabf3b4aa5000674a28e898c68cf573e381e181764a57c7452206c2df6`）、`evidence-manifest-v1.json`（SHA-256 `852eb695ec57d8593ece8c065a4a474f5b6721afa48347685c4bca89e642d929`、97 files）。raw model response、rendered prompts、tool events、command、patch、grader receipt、resource samples、protocol invalidation recordsを保存した。candidate workspace配下のnested `.git`は公開evidenceに含めない。

## 4. Progress-aware navigationと編集後終了

Sprint VIIの消費済みTRL #7913 taskのraw event replayでは、14 schema-valid actions中11 search、10 duplicate-query rejection、edit 0だった。最初のsource readの後に検索を続けた。Sprint VIIIの外部attemptでも、独立した同一query反復が7回観測された。既存のguidanceは同一queryを避けるよう伝えていたが、その文章だけではモデル動作を制御できなかった。navigation v1/v2のA/B/Cを同じfresh taskで比較する試験や、no-progress controllerの因果比較は今回実施していない。消費済み課題を再試行して効果を主張しない。

別の合成fixtureでpost-edit behaviorを3 arm比較した。protocol SHA-256 `47ab9f75b5d85ec66f36d05f9ebc25fae7e8595bf8b930b63b80b17888fe21f9`。Qwen2.5-Coder-7B-Instruct-4bit、同一fixture、greedy設定。3 arm全てでモデル生成editが適用され、独立AST graderがPASS。現行controllerは8 calls / 54 generated tokens / 120.5秒でedit後もlistを繰返し、turn budget exhausted。post-edit inventory gateは6 calls / 56 tokens / 91.9秒でmodel `finish`。gate + 明示guidanceは5 calls / 50 tokens / 81.6秒でmodel `finish`。3 armともpeak process RSSは約4.74 GB。

これは合成fixtureの診断である。モデルによる正しい編集と終了動作が実装上可能であることを示す一方、1 fixtureでの結果なので成功確率や一般的な編集能力を示さない。controllerがepisodeを強制終了した成功例ではなく、2 armはモデル自身が`finish`を呼んだ。

## 5. Long-horizon、RSI、学習

- **実モデル接続long-horizon recovery:** 未実施。外部軌跡は最大10 tool callでedit前に終了し、crash/restart injection、状態復元、重複side effectは測っていない。既存のsynthetic recovery記録はsynthetic evidenceのまま。
- **RSI research-selection loop:** 未実施。duplicate-search failureから次の仮説候補は得たが、task選択介入とrandom/fixed比較を実行していない。
- **SFT/GRPO/RL:** 未実施。実タスクの正のreward trajectoryがなく、held-out eligibilityもこのSprintで再認定していない。model updateを実行しない。
- **Held-out:** 未実施。今回のTransformers familyはDevelopmentに消費済みで、pretraining exposureもUNKNOWN。

## 6. 結論と判定

| 項目 | 判定 | 根拠 |
|---|---|---|
| Runtime source snapshot / identity audit | PARTIAL / GO for preserved replay | 97-file evidence bundle、source hash matchとaudit PASS。ただしdirty checkoutとrunner bytesのGit履歴外状態は明示的に残る。 |
| Schema-constrained tool calls | RUN / PARTIAL | 外部armで10/10 valid calls。検索進捗は制御できなかった。 |
| 新規外部Development task qualification | RUN / GO for this one Development family | baseline fail / exact fix pass / wrong mutant fail。held-outではない。 |
| 外部taskの独立修復 | RUN / NO-GO | 0/1 family verified repair。編集なし。localized armはrunnerで失敗。 |
| 合成post-edit終了診断 | RUN / PARTIAL | 3/3 synthetic edits graded PASS、2/3 model-finished。一般性能とはみなさない。 |
| Long-horizon model recovery | NOT RUN / NO-GO | crash/restartを含むreal-model trajectoryなし。 |
| RSI intervention selection | NOT RUN / NO-GO | 比較可能なtask-selection loopなし。 |
| SFT/GRPO/held-out improvement | NOT RUN / NO-GO | train signal、独立task pool、held-out gate不足。 |
| mainline merge | 0 / BLOCKED | 独立approval不足。branch protectionを維持。 |

Sprint VIIIが新たに示したものは、厳格なtool schemaを通るモデルでも、duplicate-query feedbackに従わずsource localizationで止まりうること、そして同一source editの後に終了まで誘導する合成controller gateが実行可能であること。実リポジトリでの独立修復、policy improvement、長期回復の証拠は得られていない。

次の費用対効果の高い実験は、今回消費したtaskを再利用せず、fresh Development family上で次の一介入だけを事前登録して比較すること：一度duplicate-queryが返った完全同一queryをrunnerが次回実行前に拒否し、元のmodel outputを保存したうえで短い代替検索または直近readへ誘導する。localized armのinterface parameterも独立に校正する。まずsynthetic/replayでdriver contractを試験し、すべてのruntime hashとresource preflightを固定してから新taskを露出させる。

## 7. 再現・証拠の場所

- protocol: `results/sprint_viii_20261028/external_transformers_49703/protocols/development_campaign_v5.json`
- 外部taskとqualification: `results/sprint_viii_20261028/external_transformers_49703/{task-v2,qualification,source,localization}`
- 完全なv5 model trajectoryとattempt outputs: `results/sprint_viii_20261028/external_transformers_49703/campaign_v5/attempts/`
- pre-inference v1–v4 rejection / exposure record: `results/sprint_viii_20261028/external_transformers_49703/protocols/` と `campaign_v5/task-exposure.jsonl`
- Sprint VII consumed-task navigation replay source: `results/sprint_vii_20261028/external_dev_trl7913/attempt_001/events.jsonl`（SHA-256 `6a2961c05918d916bf9f4e9d1c689347bddba3bb397d30f45bed773eeb3b22e6`）
- synthetic post-edit campaign: `results/sprint_viii_20261028/post_edit_completion_diagnostic_v5/`
- independent graders: `scripts/sprint_viii_transformers_49703_grader_v1.py`, `scripts/sprint_viii_synthetic_grader_v1.py`
- Python 3.13 evidence-only CI failure and portability fix record: `results/sprint_viii_20261028/ci/python-3.13-failure-v1.json`, raw log `python-3.13-failure-v1.log`
- relevant tests: `tests/test_sprint_viii_transformers_49703_grader.py`, `tests/test_sprint_viii_synthetic_grader.py`, `tests/test_sprint_viii_campaign_exposure.py`, `tests/test_agent_runtime_snapshot_v1.py`, `tests/test_post_edit_navigation_gate.py`

実験済みDevelopment familyを同条件で再実行してはいけない。以後の利用は非盲検診断として明示する。
