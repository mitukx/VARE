# 実モデル Coding Agent ナビゲーション診断

## 既存トレースから確認した失敗

- **#13995:** 8/8軌跡が報酬0。各軌跡で `class JsonValue` の検索が1回空振りし、その後15ターンを使って20行ページをファイル先頭から連続取得した。合計は検索8回、読取120回、編集・提出0回。対象は型エイリアス `JsonValue = ...` で、当時の検索結果に宣言候補や関連ソース本文はなかった。GRPO更新は0回。
- **#14115:** 8/8軌跡が信頼済み報酬0。128ターン中、検索127回がすべて空振り、読取は1回だけ。`defer_build`、`DeferredBuildState`、`_DeferredBuild` など、ソースにない表記を変えながら検索を続けた。記録上、検索ループの遮断・リダイレクトは0回。トークンIDとbehavior log-probabilityは2,028件で一致し、非有限値は0。編集・提出・GRPO更新は0。
- **#13994:** 8/8軌跡が報酬0。検索28回（空振り13回）、読取100回、編集・提出0回。1軌跡では `get_model_type_hints` を10回検索し、`_generics.py` の広い範囲を読み続けた。タスクに関係する `create_generic_submodel` は104行目だった。behavior log-probability 3,964件は有限だったが、報酬分散がなく更新は0回。

これらの記録は、チャットテンプレートやJSONツール形式の破損を主因とする証拠を示していない。モデル出力はネストしたJSON呼出しとして記録され、トークンとbehavior log-probabilityも整合している。一方で、検索結果から対象宣言への移動、反復検索の停止、読み取り文脈の維持が失敗していた。したがって、観測された主な障害は**モデルのタスク遂行不足と、当時のcontrollerが検索・文脈移動を制御できなかったことの組合せ**と判断した。

## 追加の小規模モデル診断

`attempt-006` は、すでに露出済みのTrain task #14115でのデバッグ専用・推論1件であり、新規評価データではない。Qwen2.5-Coder-7B-Instruct-4bitをM1 Pro 32 GiB上のMLXで実行した。16ターン、生成1,095 token、有限behavior log-probability 1,095件。検索8回、読取3回、編集ツール呼出し4回のうち適用された編集は1回、コンパイル1回、提出0回だった。適用差分は `__getattr__` の戻り値注釈を `None` から `Any` に変えただけで、信頼済みgraderは報酬0を返した。選択テストは依然失敗し、`MockValSer.__getattr__` が未初期化slotへの `hasattr(..., "__setstate__")` で再帰する `RecursionError` が記録された。OOM・タイムアウトはなかった。

この結果は、有効な編集呼出しを1回得たものの、正しい修正・提出には至らず、成功率改善を示さない。プロトコルの停止条件に従い、同じ露出済みタスクでの追加推論は行っていない。

## 実装・検証

runnerのナビゲーションを修正し、タスク文脈からAST宣言を順位付けして完全な宣言範囲を返し、読み取り済み範囲を保持し、同一編集対象の連続空振り検索を3回で制御する。自然文クエリから任意の単語を拾うfallbackも削り、短い識別子または `class` / `def` 付き識別子だけをfallback対象とした。これにより、自然文中の `Pydantic` が無関係な `PydanticErrorCodes` に一致する誤誘導を防ぐ。

実 #14115 の最初の軌跡に記録された15検索と1読取を現行検索guardへ決定論的に再生すると、空振り検索の実行は4回、遮断は11回となった（3回実行後、読取でguardを解除し、残る検索1回を実行）。これはモデルを再実行した結果ではない。

検証結果: `tests/test_local_real_model_tool_adapter.py` 26件成功、runner・navigation・testの`py_compile`成功、`git diff --check`成功。実モデル診断は変更前のv15 adapterで実行済みであり、今回のv16 fallback修正後にモデル推論した結果ではない。

## プロトコル境界

凍結済み `protocols/mlsys_actual_model_update_no_go_v2.lock.json`（SHA-256 `f91744b7e813ccaaff3396a63c639a63b8c364c61a97b361093a7b5b1c991ce4`）を変更していない。判定はNO-GO、適格なheld-out taskは0/15であり、ゲート失敗中の新規rollout・trainingは禁止されている。この作業ではoptimizer更新、SFT/GRPO、checkpoint保存・再読込、Development評価、held-outアクセスを行っていない。#13825は資格審査済みのTrain候補だが、未割当・未露出のままであり、この結果で学習許可を推定しない。
