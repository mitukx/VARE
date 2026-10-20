# 4B MLX MLSys Coding Agent Pilot — 2026-10-20

> **Evidence status:** The original PR #89 report states a 2/5 result, but the PR stack does not contain the pilot's raw trajectories, candidate snapshots, grader outputs, or hash manifest, and the cited local evidence directory was not present during the independent audit. The 2/5 figures below are therefore historical author-reported values, not independently reproducible evidence. This report does not use them to support model capability or policy-improvement claims. See the [protocol deviation and evidence audit](4b-mlx-coding-agent-pilot-v6-audit.md).

## 目的と仮説

M1 Pro上の4B級ローカルモデルが、凍結したMLSys開発タスクを調査し、コードを変更し、公開テストを通し、VAREの隔離graderで独立確認できるかを調べた。今回の仮説は「少なくとも1つの非自明な開発タスクで、モデル自身が有効な変更を作り、独立graderを通過する」である。VAREの比較実験や学習効果は今回の仮説に含まない。

## 固定条件

- モデル: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; base `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; Apache-2.0。
- 実行: Apple M1 Pro / 32 GiB、MLX 0.32.2、Colima、network-disabled candidate/grader containers、Python image `python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`。有料API・GPUクラウドは未使用。
- データ: PR #84の固定task pack commit `27955442317cf9b7beef3583b4f1a5f2662e0390`。train/development 5タスク、seed 101、temperature 0、最大12 turns、各turn最大768 tokens。4 held-outタスクは開かず、実行しなかった。
- grader: agent終了後のsource-only snapshotを、候補から隔離したVARE graderが評価。PR #84 evaluatorのSHA-256を記録。grader adapterの初期版で2つの実装問題が見つかったため、同一の保存済みcandidateを修正済みgraderで再評価した。モデル推論は再実行していない。

## 当時の報告値（未検証）

- **原報告は独立grader合格を2/5タスク (40%) としているが、raw trajectoryとgrader出力を監査できないため、結果は検証されていない。** gradient accumulation tailとwarmup cosineが合格したという記述、Wilson区間、およびStage Aゲート達成の記述は、再現可能な結果として扱わない。
- paired shuffle loader: モデルはtool callsを試したがソースを変更できず、retry上限で終了。
- row log-softmax: patchと公開テストは通過したが、独立graderの極端値ケースで失敗。
- range sums: patchは正しい出力を返したが、performance graderの2x昇格基準を満たさなかった。7組の測定でbaseline平均0.340秒、候補平均0.985秒、baseline/candidate比0.345x。よって「性能改善」としては不合格で、今回の実測は候補が遅い。
- 4/5でpatchが適用され、4/5で公開テストが通過した。残る1件は編集なしのため公開テストを走らせていない。
- 合計生成token数4,613、タスク当たりagent wall timeの中央値41.0秒、合計256.3秒。モデルプロセスの記録peak RSSは2.63 GB。host samplerでは利用可能memoryの最低値5.32 GB、free swap最低値1.18 GB。OOMや実験中断はなかった。

## Harness原因と修正

過去の`no_source_change` / `no_tool_call_retries_exhausted`について、4B実行では少なくとも一部はツール利用不能ではなく、モデルが複数turnを使っても編集toolを返さなかったことをtrajectoryで確認した。一方、全タスクで編集率100%ではないため、harnessが全原因を解消したとは言わない。

初回posthoc graderはperformance baselineの関数名を誤って公開し、range-sumsを`missing_result`にした。名前をタスクAPI名に揃えた後、JSON transportでquery pairのPython tupleがlistになる境界問題が判明した。隔離grader内で凍結したtuple pathを復元し、同じcandidateを再評価した。修正後は全7測定で出力一致し、時間の比較は有効になった。ここで初めて性能が遅いという結果が得られた。これらはgrader-only修正で、candidate patchやモデル出力は変更していない。

## 限界と結論

現在監査可能な材料だけでは、4B agentのgrader成功率を確定できない。protocolの試行数定義と実行設定にも逸脱があるため、このpilotからモデル能力、held-out一般化、VARE対基本agentの優位性、障害復旧効果、長時間実験の成功、RL/SFT/GRPOの効果を主張しない。性能タスクで候補が遅かったという原報告も、保存されたraw測定値がないため未検証である。

次段階は、追加の独立タスクとtrain/dev/held-out分割をgraderを実行する前に固定し、同一予算のbasic-agent/VARE armsとfault-injection条件を別protocolで事前登録してから実行すること。現在のheld-out 4タスクを本pilot後の調整に使わない。
