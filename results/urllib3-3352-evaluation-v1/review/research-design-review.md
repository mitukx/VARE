## 独立レビュー結果

**判定：** urllib3 #3352 は、VAREのローカル評価経路を校正する **development-only task** として条件付きGoです。実モデルの能力改善、一般化、新規研究仮説の証拠としてはNo-Goです。学習開始もNo-Goです。

静的レビューのみ実施しました。コード変更、テスト実行、モデル推論はしていません。この報告は独立したAI subagentレビューであり、人間レビューではありません。

### 対象と入力の指紋

モデルに渡す設定は、`prompt.txt` と `source_context.txt` の連結です。前者は task の症状と制約を説明し、後者は pinned base の `connectionpool.py` 抜粋です。promptに issue/PR URL、テストベクトル、gold patch は含まれていません。

- Prompt SHA-256: `110304935a1b643c21db2927534401acf2e6bcece1be21c86bcda5924a159270`
- Source-context SHA-256: `2512623c6340fdd5a3b98efbe8512d2fe481e0816129e26851d07cb0157d71e3`
- Task brief SHA-256: `299b6fde8da4069f18202c7e03e5db8c1053a1776c915965ee35546886ec2be7`
- レビュー対象のローカル VARE HEAD: `b3204926b3cae291f7c13e3134444c71a3a63172`
- レビュー入力と参照URL一覧の canonical manifest SHA-256: `f5bef3ce02629bf0b4c2fac5ca27d0cb4393486ddbedb9c873ce95eb3484be39`

この manifest fingerprint は、上記3ファイルのハッシュ、VARE HEAD、および以下の参照URLを改行で連結して計算しました。モデル出力のfingerprintではありません。

### タスク適合性と汚染

urllib3 issue #3352 は2024年2月に公開され、報告者は二重スラッシュを含む path が再解析され、`:h` 部分で失敗する症状を具体的に示しています。PR #4952 は2026年4月にマージされ、maintainerはpathで始まる入力について不要な再解析を省く説明をしています。そのため、基準コードと既知修正を再現する校正材料としては良い候補ですが、issue本文と修正は公開済みで、モデルの学習データに含まれた可能性を排除できません。[issue #3352](https://github.com/urllib3/urllib3/issues/3352), [PR #4952](https://github.com/urllib3/urllib3/pull/4952), [修正commit](https://github.com/urllib3/urllib3/commit/d150cea52b37f4633a0dff4e26339c51d23d987b)

既知修正は校正にのみ使用し、confirmation評価へ流用しないというロックの扱いは適切です。baseとfixのコードをモデル入力から分離し、screen scriptがprompt/contextだけをモデルへ渡す範囲では、task/test/goldの分離も妥当です。なお、task sidecarには公開issue/PRへの参照とcalibration revisionが記載されています。将来、ファイル閲覧やWeb閲覧を行えるagentで評価する場合は、画面に渡す入力だけでなく、その実行環境からsidecarや外部solutionへのアクセスも遮断する必要があります。

このpromptは関連ソース抜粋付きの一問一答型patch生成です。repo探索、テストを使った修正、tool-use、長いcoding-agent trajectoryは評価しません。また、upstream maintainerはredirect時の同種のpathの扱いを別問題として残しています。したがって成功を「urllib3 URL処理全般の修正」と拡大解釈せず、ロックされた直接requestの挙動に限定すべきです。

### データ・ライセンス比較

- **urllib3 #3352:** pinned source treeの `LICENSE.txt` はMITです。これはurllib3のコードと許諾対象のファイルに関する確認で、報告者がissueに投稿した文章をMITで再配布できる根拠にはなりません。タスク文は症状の要約として扱い、issue本文の長い転載を避け、出典を示すのが適切です。GitHubの公開コンテンツ規約もissue本文にMITソフトウェアライセンスを自動適用するものではありません。[pinned LICENSE](https://raw.githubusercontent.com/urllib3/urllib3/166f66faa5ef23e255f654e8c6b66ad8590fe630/LICENSE.txt), [GitHub Terms](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service)
- **SWE-bench:** データフィールドにproblem statement、base commit、gold `patch`、`test_patch`、FAIL_TO_PASS/PASS_TO_PASSが含まれます。公式submission checklistはtest知識の使用とsolutionを探すWeb閲覧に注意を促しています。正式ベンチマークとしては豊富ですが、公開solutionと実行環境があり、今回の軽量なmacOS校正にそのまま持ち込むには適しません。[dataset guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md), [evaluation checklist](https://github.com/swe-bench/experiments/blob/main/checklist.md)
- **SWE-rebench V2:** 現在のdataset cardは32,079件と `patch` / `test_patch` 等の項目を記載し、dataset licenseをCC-BY-4.0、各rowに由来repoのlicenseを別途記録しています。dataset licenseだけで個々のrepoコードやissue本文の権利が一括解決するとは扱えません。多数のtaskを候補抽出する価値はありますが、公開修正とtest patchを含むため、そのまま確認用holdoutにはできません。[SWE-rebench V2 README](https://huggingface.co/datasets/nebius/SWE-rebench-V2/blob/main/README.md)
- **Django historical bugs:** #33899はSQLite 3.35.5以降でindexed fieldを削除するmigration不具合で、issueにminimal exampleとregression testがありました。#33992はExists/subquery/aggregationとORM/databaseの組み合わせを含み、相対的に複雑です。どちらも公開済みのhistorical issueであり、確認データではなく開発用候補です。[Django #33899](https://code.djangoproject.com/ticket/33899), [Django #33992](https://code.djangoproject.com/ticket/33992)
- **代替データ:** Defects4Jは再現条件としてJava 11等を指定し、今回のPython/macOS小規模経路より環境負荷が高いです。BugsInPyではリポジトリ自身のlicenseが不明瞭だとする未解決issueがあるため、明示確認前の再配布候補にすべきではありません。[Defects4J README](https://github.com/rjust/defects4j/blob/master/README.md), [BugsInPy license issue](https://github.com/soarsmu/BugsInPy/issues/94)

候補モデルのHF表示はApache-2.0です。これはモデル配布物のライセンス確認であり、taskデータの権利やtraining-corpusの来歴・汚染状況を保証しません。[Qwen pinned license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE), [TinyLlama pinned revision](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6)

### 1 task × 1 attempt の解釈

各モデルのgreedy出力がロックされたgraderを通れば、その設定での**一回のpatch生成が成功した**とは言えます。失敗すれば、その一回が失敗したとは言えます。どちらもモデルの成功率、task群への一般化、seed間変動、学習介入の効果は推定しません。greedy設定なのでseedの反復も、単独では確率的成功率の推定になりません。

従ってこのscreenで可能なのは、生成形式、単一taskの実行可能性、graderへの接続確認までです。SFT/no-update/VAREを比較する実験根拠にはなりません。

### 懸念と次のGate

1. `protocol.lock.json` はメモリ制限が強制されないことを明記しています。wall timeout、CPU、出力制限は記載されていますが、任意の候補コードを「完全にresource-bounded」と表現するのは不正確です。少なくともOOMリスクを明記し、任意コードを広く受け付ける前に実行時メモリ境界を追加してください。
2. 現在の一問は、model feasibilityとgrader校正の開発taskとして限定してください。成功してもVAREによる改善の証拠ではありません。
3. 学習前には、未使用かつ権利確認済みのtask群、事前固定されたtrain/dev/confirmation分割、独立grader、baseの成功・報酬分散、matched no-update/SFT/VARE条件を揃えてください。公開issue/patch由来の汚染リスクを記録し、モデルごとの成功差を検出するpowerを、最小実務効果とtask間分散から事前に算定してください。confirmationを見てpromptや成功基準を調整しないことが必要です。

## 参照したURL

- https://github.com/urllib3/urllib3/issues/3352
- https://github.com/urllib3/urllib3/pull/4952
- https://github.com/urllib3/urllib3/commit/d150cea52b37f4633a0dff4e26339c51d23d987b
- https://raw.githubusercontent.com/urllib3/urllib3/166f66faa5ef23e255f654e8c6b66ad8590fe630/LICENSE.txt
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE
- https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6
- https://huggingface.co/datasets/nebius/SWE-rebench-V2/blob/main/README.md
- https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md
- https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md
- https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md
- https://github.com/swe-bench/experiments/blob/main/checklist.md
- https://github.com/microsoft/SWE-bench-Live/blob/main/README.md
- https://code.djangoproject.com/ticket/33899
- https://code.djangoproject.com/ticket/33992
- https://github.com/rjust/defects4j/blob/master/README.md
- https://github.com/soarsmu/BugsInPy/issues/94
- https://docs.github.com/en/site-policy/github-terms/github-terms-of-service

**Review prompt:** 親エージェントから指定された独立した研究設計・データ適合性レビュー依頼。対象task、base/fix revision、モデルrevision、development-only条件を確認し、比較候補、ライセンス、汚染、分離、1 task × 1 attemptの推論限界、次の学習Gateについて証拠に基づくGo/No-Goを評価しました。 **Reviewed output:** 上記レビュー要約。Fingerprintは `f5bef3ce02629bf0b4c2fac5ca27d0cb4393486ddbedb9c873ce95eb3484be39` です。
