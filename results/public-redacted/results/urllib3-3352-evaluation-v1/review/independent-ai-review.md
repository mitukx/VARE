## 独立AIレビュー結果

対象HEADは `b3204926b3cae291f7c13e3134444c71a3a63172`。レビュー対象ファイルのSHA256は次のとおりです。

- `grade.py`: `ec9ec2befbd25094e08577b32f8e37f0eff4d932ce3431353e3cfd0e29f46efb`
- `protocol.lock.json`: `6411d656a88814b496db46eaa60086b8b34abacef0ee2dc625041e7cbab688ad`
- `adversarial_test_urllib3_3352.py`: `1ef1dae59f98fed8dadead1ba980863ffc6f6c945b7cbe7f36e36c3459706d42`

**確認結果:** 旧import-time direct-socket/status-spoof攻撃は拒否されました。graderは候補コードの実行前に停止し、`candidate_source_scope_valid=false`、理由は「`HTTPConnectionPool.urlopen`外の構文変更」でした。ケースは実行されていません。既知修正は実際のurllib3経路で成功し、F2P 2件・P2P 3件すべて通過しました。slow-drip deadlineの回帰テストも通過しました。

再現コマンド（reviewerが使用した実パスから、別環境では各run rootを置き換えてください）：

```sh
PYTHONPATH=src <HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 scripts/grade_urllib3_3352_task.py \
  --workspace /tmp/vare-review-reward-exploit/workspace \
  --source-record /tmp/vare-review-reward-exploit/source-record.json \
  --task-id urllib3-preserve-ambiguous-request-target-v1 \
  --task-python <HOMEBREW_PREFIX>/bin/python3.11
```

```sh
PYTHONPATH=src <HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 scripts/grade_urllib3_3352_task.py \
  --workspace <known-fix-run>/preparation/workspace \
  --source-record <known-fix-run>/preparation/source-record.json \
  --task-id urllib3-preserve-ambiguous-request-target-v1 \
  --task-python <HOMEBREW_PREFIX>/bin/python3.11
```

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <HOMEBREW_PREFIX>/Caskroom/miniforge/base/bin/python3.12 \
  -m pytest -q tests/test_urllib3_3352_evaluator.py
```

攻撃結果JSONのSHA256は `e0aa85645a079e5b5151bcead6e70ce7f0d3274c781f3db1f8ec10bbdac68c8c`、既知修正の結果JSONは `aa8dd917a8d272fafd5c7cf19aca75de5a5643428f2bb60ef0e796b2bce55af7` です。

`candidate_scope_valid` は対象メソッド内でもURL正規化の限定ASTだけを許可し、import・任意I/O・任意コード生成につながる構文を拒否します。socket監査は許可された接続先に加えてurllib3の必要なcode objectを実行スタックに要求します。今回、現行版で対象メソッド限定の攻撃が成功するケースは確認できませんでした。slow-dripテストはreader helperの単体検証であり、悪意ある候補を通すgrader統合試験ではありません。

残る制約として、protocol lock自体は署名済みの外部アンカーではありません。候補プロセスによる実行中改変はsandboxで阻止されますが、実験公開時はgrader・lockのcommitとハッシュを外部証拠として固定してください。

これは独立した**AIレビュー**であり、人間によるレビューではありません。ファイル変更・コミット・モデル実行は行っていません。
