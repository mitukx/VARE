# Sprint IV — 外部コード修復の実験結果

**基準日:** 2026-10-27  
**VARE main:** `f3ab90a36e8724441b6b9555255f8f75a0a4d529`  
**実験実装の基点:** PR [#139](https://github.com/mitukx/VARE/pull/139) head `087914fb9fc127302ab632cc665f9b7352b0c47c`  
**凍結 protocol v2:** SHA-256 `b208be82c3f864b701c4d35286fb417b4e219c6b492d496b2bad7b2b147a3308`

## 新しく得た結果

Qwen2.5-Coder-14B-Instruct-4bit を MLX CPU で使い、外部 MLX-LM の新しい Development task family を2件、各4回、計8回実行した。source-localized 条件では Falcon-H1 の `__call__` に2回、モデル自身がレイヤーごとの空 cache 初期化を加え、双方が公開テストと独立 hidden grader を通過した。2回は同一task family内のseed違いであり、独立修復数は2件ではなく **成功したtask family 1/2** と報告する。

自律ナビゲーション条件は4回とも編集に至らなかった。MiniCPM3 task は source-localized で2回編集され、公開テストを通過したが、独立hidden testを2回とも失敗した。この対照により公開テストだけでは誤修復を受け入れることがわかった。広いcoding能力や学習後の改善を主張する標本数ではない。

## 実験計画と修正履歴

最初のv1 cohortは、候補可視の `TASK.md` / `task.json` に既知の修正操作が含まれると事前実行監査で判明したため、**推論前に無効化**した。protocol hashは `3909cb938df4d87c09060afa6f56dbd3a4c5b392dd7b3d5deedc9d35f50cf98f`、起動したモデル試行は0件。旧protocolや監査記録は書き換えず、異なるtask IDと行動記述のv2を作成してから実行した。

v2の主指標は固定4試行/ task family（2 arms × 2 seeds）で「独立graderに合格した修復を1件以上含むDevelopment task familyの割合」。比較は自律ナビゲーションAと、凍結lexical localizationで関数を提示するB。両条件とも同じモデル、サンプリング設定、編集lease、テストgraderを使った。seedは431/432、temperature 0.2、top-p 0.95、top-k 20、1ターン512生成token、1試行2048生成token、12ターン、8192 context token、900秒上限。結果を見てから課題・予算・判定基準は変更していない。

| Task family / task ID | Upstream base revision | Known-fix revision | 対象 |
|---|---|---|---|
| [MLX-LM #1926](https://github.com/ml-explore/mlx-lm/pull/1926), `mlx-lm-1926-falcon-h1-uncached-behavior-v2` | `87b7b583a697537aa68f47130b40884700b5f55f` | `ebba55f5176fe451b94f04212e975f052a6d96e0` | `mlx_lm/models/falcon_h1.py`, `FalconH1Model.__call__` |
| [MLX-LM #1913](https://github.com/ml-explore/mlx-lm/pull/1913), `mlx-lm-1913-minicpm3-left-padding-behavior-v2` | `c69d1288440a0dc4e6401fc417098b07598dccd5` | `b4844fd8072b1fc241ce3bb5f81526346b80370f` | `mlx_lm/models/minicpm3.py`, `MiniCPM3Model.__call__` |

両課題とも候補workspaceにはbaselineとpublic testだけを置き、hidden graderを候補プロセスから隔離した。別Docker workspaceでのbaseline / known-fix / incorrect-mutant controlsを実施した。MLX-LM PR #1926 taskはbaselineがpublic/hidden fail、known fixがpass、mutantがfail。PR #1913 taskはbaselineとmutantがpublic smokeを通る一方hidden fail、known fixだけpublic/hidden passであり、public testの弱さも検出した。モデル事前学習でこれらのソースや修正に接触していないとは言えない。VAREの過去task記録には一致する試行がなく、pretraining exposureは不明である。凍結protocolのsynthetic prerequisiteはこの評価とは別に、実モデルの4/4 tool call受理とsynthetic public/hidden passを要求しており、外部repair数には加算していない。

## モデルと計算資源

- **Checkpoint:** `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
- **Snapshot:** `29efdbab55a161237ab1e432a3abaf6c7ae2b477`、4-bit
- **Backend:** MLX 0.32.4 / mlx-lm 0.32.2、Apple M1 Pro CPU、Python 3.12.12。既存cacheを使用し、重みは取得・変更・Git追加していない。
- **Tokenizer:** tokenizer JSON SHA-256 `a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf`、config SHA-256 `7e88129d9769a0b14b1587a7d5e829fe93ac0e1511636471fdfc0811951418e6`。
- **Grader:** Docker 29.5.2 / Colima / `vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7`。network none、read-only root、1 CPU、2 GiB grader memory、32 pids。
- **凍結済み資源上限:** 試行900秒、process RSS 8192 MiB、swap growth 160 MiB、開始free swap 380 MiB以上、試行中free swap 220 MiB以上。実開始free swapは425.12 MiB。上限を引き上げていない。

8試行すべてresource abortは0。最大経過時間368.436秒、最大process RSS 6835.828 MiB（約6.68 GiB）、swap増加0 MiB、最小free swap 425.12 MiB。合計wall timeは1792.547秒、prompt token 138,422、generated token 9,223。並列で大規模モデルを実行していない。

## 全試行

`parse` は出力のうちstrict tool-call parserが受理した数、`read/edit` は読取数/実際に適用された編集数。grader欄はpublic/hiddenの順。token欄はprompt/generated。

| Task family / PR | Arm | Seed | Parse | read/edit | finish | Grader | Wall秒 | Peak RSS MiB | Tokens in/out |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon-H1 / #1926 | Autonomous | 432 | 3/6 | 1/0 | no | 未採点 | 310.064 | 2998.594 | 27140/1641 |
| Falcon-H1 / #1926 | Localized | 432 | 3/3 | 1/1 | yes | pass/pass | 66.434 | 6835.828 | 5917/280 |
| Falcon-H1 / #1926 | Autonomous | 431 | 2/5 | 1/0 | no | 未採点 | 233.545 | 5877.094 | 17713/1607 |
| Falcon-H1 / #1926 | Localized | 431 | 3/3 | 1/1 | yes | pass/pass | 84.633 | 4037.766 | 5933/295 |
| MiniCPM3 / #1913 | Localized | 431 | 3/6 | 1/1 | no | pass/fail | 356.664 | 5263.250 | 15931/1749 |
| MiniCPM3 / #1913 | Autonomous | 431 | 5/8 | 2/0 | no | 未採点 | 368.436 | 3740.938 | 33527/1724 |
| MiniCPM3 / #1913 | Autonomous | 432 | 4/7 | 2/0 | no | 未採点 | 304.042 | 1456.328 | 26084/1690 |
| MiniCPM3 / #1913 | Localized | 432 | 3/3 | 1/1 | yes | pass/fail | 68.729 | 6365.891 | 6177/237 |

全体で41出力中26（63.4%）をstrict parserが受理し、tool call 26件、function read 10件、適用編集4件、tool-level finish 3件だった。Autonomousは4/4編集0、Localizedは4/4編集が適用され3/4でfinish。独立graderのpublic passは4件、hidden passは2件、正しい修復は2試行だがFalcon-H1の1 familyに集中した。

1/2 familiesの比率に対する二項Clopper–Pearson 95%区間は約0.013–0.987。paired arm比較も2 familyのみ（FalconでLocalizedのみ成功、MiniCPM3は両arm失敗）なので有意なモデル間・controller間差は結論できない。これは独立に採点された修復の成立を示す実行可能性証拠であり、広いbenchmark能力の推定ではない。

## Runner障害と回帰修正

全8 subprocessはモデル/ツール相互作用の後、最終記録を書き出す際に非0終了した。runnerが `git diff <external-base-sha>` を呼び、外部task revisionがVARE側Git object databaseに存在しないため失敗し、`agent.json` と `run_finished` eventが作られなかった。これはresource abortではない。各イベントhash chainは検証済みで、2件のFalcon試行では適用編集、tool-level `finish`, 別workspaceでのpublic/hidden PASSがそろう。MiniCPM3はhidden failのため修復として数えない。

回帰修正は、推論前に候補可視ソースの正確なbytesを保存し、最終patchをそのbytesとの差分で生成する。既知のexternal revisionをVARE Gitに偽装・追加せず、source hash / edit lease / grader validationは変更しない。実際に失敗したPR #139コード経路に対する独立PRは [#141](https://github.com/mitukx/VARE/pull/141)、commit `52d257e80f503306d04af255203fef9396b4d304`。changed/unchanged sourceをGit metadataなしで出力するregression testsを含み、関係suiteは105 passed、Ruff / `git diff --check`もpassした。PR #141は#139をbaseに積まれており、mainには未統合。v2のモデルを再実行していない。

公開証拠bundleは `results/sprint_iv_20261027/external-repair-evidence-public-v1/` にあり、8つのraw hash-chained event journal、候補diff、resource samples、正規化したcommands、grader control summaries、protocol、task/public test、manifest verifierを収録する。hidden test source、hidden grader、hidden stdout/stderr、candidate workspaces、model weightsは公開bundleに含めない。未編集の生記録は元のローカル実験treeに残す。

## 学習・long-horizon・RSI

SFT / GRPOは**未実施**。二つともDevelopment taskでTrain例ではなく、修復をtraining dataとして流用しない。15-family held-out gateを変更せず、held-out taskを消費していない。optimizer update / training checkpoint / learned policy comparisonはない。

Model-coupled asynchronous RL、optimizer recoveryとの接続、独立したlong-horizon fault injection、agent提案のRSI benchmark loopも**未実施**。今回の完了は1つのbounded coding episodeに限られ、504-round synthetic systems結果の再現や延長ではない。

## PRと統合状態

mainは `f3ab90a36e8724441b6b9555255f8f75a0a4d529` のまま。PR #139はDraft、head `087914fb…`、base PR #137の `6d95bd91…`。#139の最初のcheck run `40408880544` でPython 3.13が失敗したが、同じheadのrerun `40408886276` では3.11/3.12/3.13を含む全checksがpassした。人間のapprovalは記録されていない。

PR #141は外部revision差分出力だけのfocused child PR（head `52d257e80f503306d04af255203fef9396b4d304`）で、main merge前に#139依存が解消される必要がある。PR #141のGitHub checksはこの報告snapshot時点で実行中で、完了済みchecksはpass、失敗checkはない。公開用の実験レポート・trajectory bundleは、mainをbaseとする別PR [#142](https://github.com/mitukx/VARE/pull/142)（head `2d519b57446401d9470212987d76315ad2a23a04`）に分けた。#142もsnapshot時点ではCI実行中。PR #140 head `a548daa1797b0d6a7a9b865f2eebb18b8ca71fc5` のCIはpassだが、symlink/archive fallbackに関するautomated P2 threadは未解決のまま独立review待ち。PR #120のgrader-completion spoofing P1も既存blockerである。

今回のPR変更履歴は開始時16 openから18 openへ増加し、#141（runner修正）と#142（実験証拠）の2件を作成した。close 0、merge 0、既存PRのDraft解除0。Draft数は8のまま。mainは`f3ab90a36e8724441b6b9555255f8f75a0a4d529`のままで、PR reviewやdependency gateを飛び越えた統合はない。

## 判定と次の実験

| 項目 | 判定 | 根拠 |
|---|---|---|
| 外部モデルのbounded tool trajectory | RUN | 8/8でhash-chain有効、14Bが41 model outputを生成 |
| Qwen 14Bが外部編集を適用 | RUN | localized 4/4 edits applied、autonomous 0/4 |
| 独立検証済み外部repair | RUN / 小規模 | Falcon-H1 2成功試行、1/2 task family |
| 自律ナビゲーション | NO-GO / controller改良必要 | 4/4で編集前に同種のnavigationを反復 |
| Training / GRPO / SFT | NOT_RUN | eligible Train dataなし |
| held-out improvement | NOT_RUN | 未使用・15 family要件維持 |
| model-coupled long-horizon RL | NOT_RUN | 今回実行していない |
| RSI research automation | NOT_RUN | agent-proposed benchmark iteration未実施 |
| runner export fix | PARTIAL | PR #141に実装・105 tests pass、CI/review待ち、main未統合 |
| PR close / merge | 0 / 0 | review・dependency gateを維持 |

**次の最優先実験:** PR #141のreview/CIを完了させてから、未使用のMLX/Transformers Development task familyを少なくとも6件、同じ14B checkpointと修正runnerで事前登録し、autonomous/localizedを同seed・同tool budgetで比較する。各runはrunner終了と`run_finished` eventまで確認してから別graderに渡す。今回のtask IDは再利用しない。学習は、その後に独立Train familyとreward variationが確認できた場合だけ始める。

## 検証

公開bundle内から:

```sh
python3 verify_evidence.py
shasum -a 256 -c SHA256SUMS
```

このコマンドは固定protocol、raw event hash chain、公開bundleファイルを検証し、モデルを実行しない。
