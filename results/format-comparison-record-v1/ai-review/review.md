# 実行者不明の既存record：独立読み取り監査

対象は `/Users/user/vare-format-vs-correctness-v2`。本監査では学習・モデルの
load・生成・bootstrapを実行していない。元runへの書込みも行っていない。
963入力fileを監査前後にSHA-256照合し、変化はなかった。

progress SHA-256:
`10860ac27af371a7374bcb7ca38c6bd5171f6d467aef43c3a7c1e95468c73694`

## 実行した検査

| 対象 | 結果 |
| --- | --- |
| record grid | 320 train / 17,920 eval。現在の固定gridとの欠落・重複なし。sample seed / train seed不一致0 |
| 整数採点 | 独立したAST + Fraction計算と独立したstrict最終行parserで全evalを再採点。現scorerとの不一致0 |
| train reward / advantage | 全240 RL群をraw outputから再計算。不一致0 |
| checkpoint | adapter/config/optimizer各320fileをNumPy + safetensorsで検査。全tensor有限、optimizer stepとdirectory不一致0 |
| 入力保全 | 全963fileの監査前後hash一致 |

確認用task expressionは現在の固定seed定義とcommit済みheldout packから
再構成した。run内に実際のpromptやexpressionは保存されていないため、
当時モデルへ渡したtask identityをこの検査だけで証明することはできない。

### 主要なraw整数正解数

| arm | checkpoint | confirm正解 / 1280 | confirm strict format / 1280 |
| --- | --- | --- | --- |
| base | 0 | 323 | 1280 |
| format SFT | 16 | 332 | 1280 |
| format RL | 8 | 329 | 1280 |
| format RL | 16 | 336 | 1280 |
| correctness RL | 8 | 329 | 1280 |
| correctness RL | 16 | 338 | 1280 |
| random RL | 16 | 336 | 1280 |

primaryの直接計数差は `(338 - 332) / 1280 = 3/640 = 0.0046875`。
既存summaryのmeanに一致した。信頼区間は独立に再計算していない。

## 切り分けられた失敗原因

### 1. format RL：報酬飽和とadapterの無更新が同時に確認された

- train 160出力はすべてstrict formatを満たし、正解は24出力だった。
- 全80群のrewardが `[1, 1]`、advantageが `[0, 0]`、lossが0。
- 全80checkpointのLoRA Bは要素がすべて0。LoRA Aは各seedのstep1から
  step16まで完全一致し、adapter tensor全体もseed内で全step同一。
- optimizerの全m/v tensorも全stepで0。step counterのみ進んでいる。

現在のgroup centeringでは等しいrewardからadvantageが0になる。現在の
MLX objectiveは `mean(advantage_i * completion_NLL_i)` であるため、この
記録はゼロの学習信号と整合する。B=0なら通常のLoRA積によるdeltaも0になる。
実MLX forwardをロードして確認した結果ではない。

format RLのstep8とstep16はadapterが同一でも正解数が329から336へ変化
している。sample seedはcheckpoint stepを含むため異なる。これらの
score差をadapterの学習効果とする説明は、このrecordからは支持できない。
凍結base weightや当時の実行経路まで保存されたrecordではない。

### 2. correctness RL：報酬差は8群、保存tensor変更は39回

rewardは `[0,0]` が61群、`[1,1]` が11群、mixedが8群。各seedの初めて
mixedになるstepと、初めてLoRA Bが非zeroになるstepが一致した。

| seed | mixed reward steps | 初B非zero step |
| --- | --- | --- |
| 20261016 | なし | なし |
| 20261017 | 6, 8 | 6 |
| 20261018 | 6, 8, 10 | 6 |
| 20261019 | 9 | 9 |
| 20261020 | 8, 9 | 8 |

72群のzero advantageを「72回parameter更新なし」と解釈してはいけない。
最初の非zero signal後はAdamのmomentumが残り、保存tensorはその後の
zero-advantage stepでも変化している。step1を基準とした隣接checkpointの
tensor変更は合計39回。seed20261016は全16stepでformat RLと完全同一だった。

### 3. random RL：mixed rewardだけでは非zero gradientを保証しない

mixed rewardは31群あるが、うち15群は2本の出力文字列が完全同一で、
recorded lossも0。同じdeterministic tokenizerで同じ出力をencodeすると、
同じNLLに正負のadvantageが掛かり相殺する。現在の設定はLoRA dropout 0。
各seedの初B非zero stepは最初のmixedかつ非同一出力のstepと一致した。

random rewardの31 mixed群を31有効更新と数える説明も支持できない。
この相殺の説明は現在のobjectiveを静的に読んだ結果であり、当時の勾配を
実行して再計算したものではない。

## 整合性の追加反証

format RLとcorrectness RLのadapterが一致する評価checkpointを照合した。
seed20261016のstep8/16とseed20261019のstep8では、同sampling seedの
1024出力すべてが文字列完全一致した。この範囲で状態と出力の矛盾はない。

## 来歴と検証範囲の限界

- 実行者不明の既存recordである。実行者を推定で認定していない。
- environmentはPython 3.12.12 / mlx_lm 0.32.0 / backend mlxのみ。
  実行source revision、source hash、MLX本体version、device、モデルweight
  revision、完全な依存関係、実行commandを確定する情報は保存されていない。
- recordのprotocol hashとmodel repo IDは現在の固定定義と一致するが、
  それだけでは使用したsourceやweightを証明しない。
- prospective analysis pinもanalysis row tagもない。v3はretrospectiveと
  して扱う必要がある。現在のv3 commitの成立時点は本監査では検証していない。
- step0 adapterがない。保存step間の完全一致とB=0を確認した。
- raw record / 保存tensorの整合性確認であり、制御実験や当時の実行完全性を
  認定する成果ではない。確認済みcohortの新たな調整利用は行っていない。

## 実行と静的確認の区別

実行したのはJSON検査、整数再採点、報酬とadvantage再計算、CPU NumPyでの
safetensors比較、hash照合である。数式の説明は現在の固定source
`3a5a808e71d948cbe531a952344f8d5dee91f625` の
`src/vare/format_vs_correctness_v2.py:151` と
`scripts/run_format_vs_correctness_comparison_v2.py:248`、`:276` を静的に
読んだ結果である。実モデルのforward/gradientを実行した確認ではない。

再実行入口:

```sh
/opt/homebrew/Caskroom/miniforge/base/bin/python3.12 /Users/user/Documents/Codex/2026-10-13/var/work/reviewer/format-result/audit_existing_record.py
/opt/homebrew/Caskroom/miniforge/base/bin/python3.12 /Users/user/Documents/Codex/2026-10-13/var/work/reviewer/format-result/audit_duplicate_rollouts.py
```

詳細は `audit.json`、`duplicate-rollouts.json`、`input-manifest.json`、
`checkpoint-manifest.json`、`audit-environment.json` に保存した。

## Portable witnessの後続review

新しいportable witnessを静的に確認し、独立parserのsingle-space限定と
strict-format照合の欠落を指摘した。今回の実record出力にはparser差はなかった。
修正 `148a429673c52da08834b9c6a9b9b60ea1913299` では両方が解決されている。
構成した6出力の限定probeで、multiple spaces/tabを含む修正後のparserと
strict-format/strict-correct判定の一致を実行確認した。原runの全件検査は
繰り返していない。

この後続確認は `portable-script-review.md` と
`portable_parser_corrected.json` に記録した。未解決の具体的な指摘はない。
correctness/random checkpointの追加検査は上記のローカル監査に基づき、
公開portable witnessが保持するformat checkpointの範囲とは区別する。
