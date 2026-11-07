# ResearchGym v1 技術報告

この報告は、実装したループが今回の CPU 実行で実際に返した記録だけを書く。将来のモデル学習や GPU 上の性能は書いていない。

## 何を動かしたか

`vare research` は、編集可能な課題ワークスペース、公開測定、隠し採点、実験ジャーナル、中断後の再開を一つの実行ディレクトリにまとめる。課題は3種類。

| 課題 | 種類 | 隠し採点が確認したこと |
| --- | --- | --- |
| `grad_accum_scale_v1` | 学習ステップの数値 | マイクロバッチ勾配の平均。空入力は `ValueError` |
| `loop_invariant_score_v1` | 同じ値を保った高速化 | 値の一致。短縮時間の 95% 下限が、A/A 絶対差の 95% 上限を超えること。固定倍率はゲートではない |
| `descent_sign_debug_v1` | 発散する実験の修復 | 非公開系列での再計算 MSE と、平均で割った1ステップ。関数が返す損失は採用しない |

既知の正解ファイルと意図的な不正ファイルは採点器の校正にだけ使う。エージェントのワークスペースにはコピーしない。

候補関数は別プロセスで動かす。親プロセスが期待値を持ち、子プロセスには渡さない。子の `PYTHONPATH` にリポジトリは入れない。編集 API はワークスペース外、絶対パス、`..`、シンボリックリンクを拒否する。タイムアウトは `timeout` であり、正解でないことの判定にはしない。

この境界は同一ユーザーのプロセスグループであり、信頼できる OS サンドボックスではない。アドレス空間のハード制限はかけていない。`HARD_ADDRESS_SPACE_LIMIT_APPLIED` は false で、`memory_limit_applied` も false のままである。`memory_mb` は記録用で、`VmHWM` の事後サンプルは割り当てを止めない。ネットワーク名前空間もない。ワーカー内の候補コードはソケットを開ける。コントローラが `shell` と `network` アクションを拒否しても、その穴は残る。合格は、ネットワークやメモリが閉じ込められたという意味ではない。

## 測定条件

再測定時の HEAD は `a9903296433a10ad553d9289ad6dba93a792940d` で、性能ゲートの述語変更自体はそのコミットの後の作業ツリーにある。課題パックのピンは `suite_sha256` `a309981b8933a0012f3f160876522c21e29c0cfd56877939ebc742416474bfc7` である。

この節の数値は、その時点のバースト計測（ベースライン列と候補列を別々に取る）の記録である。その後、同じ述語 `gain_low > noise_floor` のまま、各インデックスをベースライン、候補、ベースラインの順に 12 回取る実装に変わった。数値は取り直していない。

実行環境は 4 vCPU の Intel Xeon ゲストである。MacBook Pro M1 Pro では測っていない。MLX も Metal も起動していない。この速度を Apple Silicon や GPU の結果として扱わない。`counts_as_autonomous_model_success` は false のままである。

生の要約は `docs/research_lab/records/` にある。ホストのパスは入れていない。

## 採点器の校正

`expectations_met` は true。既知の正解だけが合格し、未修正と意図的な不正は不合格だった。

勾配蓄積の最大絶対誤差は、未修正 0.224、不正な追加スケール 0.267、既知の正解 5.55e-17。既知の正解とは別の等価な実装も単体テストで合格しており、採点はソース文字列の一致ではない。

スコア関数の未修正は値の誤差 1.11e-16、記述用の速度比 0.989、短縮時間の下限 -0.00085 秒、A/A ノイズ上限 0.00125 秒で不合格。既知の正解は誤差 1.11e-16、記述用の速度比 8.85、短縮時間の下限 0.00664 秒がノイズ上限 0.00348 秒を超えて合格。オフバイワンの平均は誤差 0.646 で、速度採点の前に値の不一致で不合格。

勾配降下の未修正は再計算 MSE 1.76e6、パラメータ誤差 1699。学習率だけを 0.025 にした不正例は MSE 231、ステップ誤差 0.486 で不合格。符号を直した既知の正解は MSE 1.09e-10、パラメータ誤差 1.85e-5、ステップ誤差 0。

ピーク常駐集合は `/proc` の `VmHWM` で、この校正では約 11–12 MB だった。

## 決定的コントローラ

`diagnostic` は学習済みモデルではない。公開測定を見て、最初の修復を一つ試し、それが公開基準を外れたら次の修復をベースラインから作り直す。

3課題とも、拒否された仮説のあとに隠し採点へ通り、`success_class` は `diagnostic_policy` だった。`counts_as_autonomous_model_success` は false。

| 課題 | 拒否 | アクション数 | 記録上の wall 合計 |
| --- | --- | --- | --- |
| 勾配蓄積 | `extra_normalization` | 4 | 0.093 s |
| スコア | `off_by_one_average` | 4 | 0.250 s |
| 勾配降下 | `shrink_learning_rate` | 4 | 0.094 s |

`first-hypothesis` は学習率を下げただけで提出し、隠し採点は不合格のまま残した。`oracle` は既知の正解を適用して採点器は通ったが、`verified` は false、`success_class` は `oracle_control`、`oracle_calibration_passed` は true。オラクル合格はエージェントの成功に数えていない。

## 外部エージェントの一連の操作

提案者は `external`。このクラウド上のコーディングエージェントが、各測定のあとで次の編集を選んだ。リポジトリ内のモデル推論ではない。`model_backed` は false。

課題は `descent_sign_debug_v1`、run id は `ext-descent`。

1. ベースライン。`diverged=true`、再計算 MSE 1.57e6、ステップ誤差 0.619。
2. 学習率を 0.25 から 0.025 にした。発散フラグは消えたが MSE は 163、ステップ誤差は 0.619 のまま。公開基準で拒否。
3. 残差を `y - pred` から `pred - y` に変えた。ステップ誤差は 0 になったが、下げた学習率が残っていたため MSE は 0.182。公開基準で拒否。
4. 学習率を 0.25 に戻した。MSE 1.37e-11、ステップ誤差 0。公開基準は支持。
5. 提出。隠し採点は合格。再計算 MSE 1.09e-10、パラメータ誤差 1.85e-5、ステップ誤差 0。

`success_class` は `external_agent_assisted`。完了後の `resume` はアクション数 5 のままで、ジャーナルハッシュは `ecca9f38e123f639b65f2e299bd8c3b266e6776e5ec2bb4c2a2526b47b3ef7e4`。エクスポートの `training_authorized` は false。

3 の拒否は重要である。符号の修正だけでは、前に入れた学習率の変更が残っていると公開 MSE が基準に届かない。決定的ポリシーは各仮説をベースラインから独立に当てるのでこの積み残しは起きない。外部セッションは起きた。

## テスト

この節の 34 件は #198/#201 時点の件数である（#196の31件に3件追加）。#200 の隔離攻撃テスト7件と、本統合ブランチの既存OSS task contract回帰1件を加えた現行コレクションは42件。分類は3課題の校正、等価実装、パストラバーサル、シンボリックリンク、偽造JSON、無限ループtimeout、壊れたworker出力、grader改ざん、隔離backend不在、grader import拒否、memory/network残存リスク、計測ノイズゲート、controller、resume、CLI、reward-free exportである。ローカルmacOSでは隔離が未確認のため、採点が必要なテストは誤答判定ではなく `isolation_unverified` を確かめる。Linuxの隔離graderでbaseline/known-good/wrong-mutantを実行した証拠はCI結果を参照する。

## 再現

```bash
PYTHONPATH=src python -m pytest tests/research_lab -q
PYTHONPATH=src python -m vare research calibrate --work-root artifacts/research_lab/calibration
PYTHONPATH=src python -m vare research run --task descent_sign_debug_v1 --run-id demo --proposer diagnostic
```

`artifacts/` は git の対象外である。残した記録は `docs/research_lab/records/`。

## 2026-11-07 統合時の回帰とOSS task境界

#202 が記した 34 件は #198/#201 の時点の件数である。#200 の隔離攻撃回帰を
含めた `tests/research_lab` は 41 件を収集し、既存の
`EngineeringTaskSpec` → `AgentTaskContract` 経路を検査する1件を加えて42件となる。
OSS課題用の別grader/loaderは作らず、このmanifest・agent contractと各課題の
hash-lock済みgrader (`mlsys_repo_task`、`mlxlm_1912_repo_task`、
`transformers_repo_task`) を再利用する。これはインターフェース確認であり、
新規課題の選定・資格審査・split割当ではない。

この時点の実行環境は Apple arm64 の macOS で、`bwrap` がなく Docker daemon も
接続できなかった。そのため統合時の校正は全9ケース（3課題×baseline/known-good/
incorrect）が `isolation_unverified` となり、候補コードを採点実行していない。結果は
[`integration-preflight-20261107.json`](records/integration-preflight-20261107.json) に保存した。
known-goodの不合格も候補の誤答とは分類せず、未評価のまま保持する。これは
ResearchGymの隔離fail-closed結果であり、正解修正の再確認にはLinux exact-head CI
で attested bubblewrap 実行が必要である。隔離を回避する同一ユーザー実行や
報酬化は行わない。

ResearchGymから出すexportは依然としてrewardを含まず、Train/Developmentへの
自動投入、optimizer更新、model rollout、Held-Out利用を許可しない。v2の0/15
NO-GOと#64830 UNSUPPORTEDの状態は不変である。

## 統合

`vare research export` は `vare.research_lab.qualification_export.v1` を出す。訓練は認可しない。報酬は作らない。held-out 分割は開かない。GRPO もオプティマイザも呼んでいない。契約は `docs/research_lab/integration_contract.md`。

## 今回示していないこと

- 学習済みモデルが自分で修復したこと。診断ポリシーは決め打ちの順序で、外部セッションは別プロセスのコーディングエージェントである。`counts_as_autonomous_model_success` は false。
- モデルの能力、または訓練してよいという判断。
- M1 Pro、Metal、GPU での速度。ここに書いた時間は Xeon ゲストの CPU 時間である。
- 信頼できる OS サンドボックス。メモリのハード上限とネットワーク遮断は未実装の穴である。
- 既存の凍結プロトコルや held-out 分割の再実行。
