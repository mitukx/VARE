# ResearchGym v1 技術報告

この報告は、実装したループが今回の CPU 実行で実際に返した記録だけを書く。将来のモデル学習や GPU 上の性能は書いていない。

## 何を動かしたか

`vare research` は、編集可能な課題ワークスペース、公開測定、隠し採点、実験ジャーナル、中断後の再開を一つの実行ディレクトリにまとめる。課題は3種類。

| 課題 | 種類 | 隠し採点が確認したこと |
| --- | --- | --- |
| `grad_accum_scale_v1` | 学習ステップの数値 | マイクロバッチ勾配の平均。空入力は `ValueError` |
| `loop_invariant_score_v1` | 同じ値を保った高速化 | 値の一致と、ベースラインに対する 2 倍以上の中央値速度。A/A の相対差が 0.75 を超える測定は不合格 |
| `descent_sign_debug_v1` | 発散する実験の修復 | 非公開系列での再計算 MSE と、平均で割った1ステップ。関数が返す損失は採用しない |

既知の正解ファイルと意図的な不正ファイルは採点器の校正にだけ使う。エージェントのワークスペースにはコピーしない。

候補関数は別プロセスで動かす。親プロセスが期待値を持ち、子プロセスには渡さない。子の `PYTHONPATH` にリポジトリは入れない。編集 API はワークスペース外、絶対パス、`..`、シンボリックリンクを拒否する。タイムアウトは `timeout` であり、正解でないことの判定にはしない。

この境界は同一ユーザーのプロセスグループである。コンテナでも別ユーザーでもない。ネットワーク名前空間もない。シェル起動アクションはコントローラが拒否する。

## 測定条件

記録は親リビジョン `85a0cfedcb602b9e1476d1bed8cf2210b59d728b` 上の作業ツリーで取った。課題パックの `suite_sha256` は `d8a5571db7867fffc1fb4233cea4e02fbdc76c7086531794de01e3d4ae5bee0f`。

実行環境は 4 vCPU の Intel Xeon ゲストである。MacBook Pro M1 Pro では測っていない。MLX も Metal も起動していない。ここからの数字を Apple Silicon や GPU の性能としては使えない。

生の要約は `docs/research_lab/records/` にある。ホストのパスは入れていない。

## 採点器の校正

`expectations_met` は true。既知の正解だけが合格し、未修正と意図的な不正は不合格だった。

勾配蓄積の最大絶対誤差は、未修正 0.224、不正な追加スケール 0.267、既知の正解 5.55e-17。既知の正解とは別の等価な実装も単体テストで合格しており、採点はソース文字列の一致ではない。

スコア関数の未修正は値の誤差 1.11e-16 で、速度はベースライン比 1.03 倍。A/A の相対差は 0.021。2 倍ゲートで不合格。既知の正解は誤差 1.11e-16、速度 6.06 倍、A/A 相対差 0.004。オフバイワンの平均は誤差 0.646 で、速度採点の前に値の不一致で不合格。

勾配降下の未修正は再計算 MSE 1.76e6、パラメータ誤差 1699。学習率だけを 0.025 にした不正例は MSE 231、ステップ誤差 0.486 で不合格。符号を直した既知の正解は MSE 1.09e-10、パラメータ誤差 1.85e-5、ステップ誤差 0。

ピーク常駐集合は `/proc` の `VmHWM` で、この校正では約 11–12 MB だった。

## 決定的コントローラ

`diagnostic` は学習済みモデルではない。公開測定を見て、最初の修復を一つ試し、それが公開基準を外れたら次の修復をベースラインから作り直す。

3課題とも、拒否された仮説のあとに隠し採点へ通り、`success_class` は `diagnostic_policy` だった。`counts_as_autonomous_model_success` は false。

| 課題 | 拒否 | アクション数 | 記録上の wall 合計 |
| --- | --- | --- | --- |
| 勾配蓄積 | `extra_normalization` | 4 | 0.093 s |
| スコア | `off_by_one_average` | 4 | 0.250 s |
| 勾配降下 | `shrink_learning_rate` | 4 | 0.093 s |

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

`python -m pytest tests/research_lab -q` は 25 件すべて合格。範囲は、3課題の校正、等価実装、パストラバーサル、シンボリックリンク、偽造 JSON、無限ループのタイムアウト、壊れたワーカー出力、時計の差し替え、採点パッケージの import 失敗、シェルアクションの拒否、診断ポリシー、最初の仮説の失敗、オラクルのラベル、ジャーナルの重複防止、末尾の壊れた行、キャンセル、CLI の再開、提出後のワークスペース改変が `stale_source` になること、報酬フィールドと凍結プロトコル ID のエクスポート拒否。

## 再現

```bash
PYTHONPATH=src python -m pytest tests/research_lab -q
PYTHONPATH=src python -m vare research calibrate --work-root artifacts/research_lab/calibration
PYTHONPATH=src python -m vare research run --task descent_sign_debug_v1 --run-id demo --proposer diagnostic
```

`artifacts/` は git の対象外である。残した記録は `docs/research_lab/records/`。

## 統合

`vare research export` は `vare.research_lab.qualification_export.v1` を出す。訓練は認可しない。報酬は作らない。held-out 分割は開かない。GRPO もオプティマイザも呼んでいない。契約は `docs/research_lab/integration_contract.md`。

## 今回示していないこと

- 学習済みモデルが自分で修復したこと。診断ポリシーは決め打ちの順序で、外部セッションは別プロセスのコーディングエージェントである。
- モデルの能力、または訓練してよいという判断。
- M1 Pro、Metal、GPU での速度。
- 敵対的なコードに対する OS サンドボックス。
- 既存の凍結プロトコルや held-out 分割の再実行。
