# RSI 実験用ローカル・プリフライト

`vare rsi preflight` は、現行の MLX/Qwen3 実行スクリプトが使うローカル環境要件を読み取り専用で確認します。モデルをロード・ダウンロードせず、候補コードを実行せず、学習も開始しません。Docker の確認には `docker info` を使いますが、コンテナーは起動しません。

## 確認項目

- Python 3.11 以上、Apple Silicon、`mlx` と `mlx_lm`
- 固定した Qwen3-4B モデルのローカル設定・重みファイル
- 長期実行と GRPO の実行スクリプト
- 空きディスク 8 GiB 以上
- 空きメモリ 8 GiB 以上、macOS のメモリ空き率 15% 以上
- 開始時スワップ使用率 94% 未満
- 利用可能な Docker デーモン

メモリやスワップを取得できない場合、その項目は `unverified` になり、全体は `incomplete` になります。閾値は現在の Qwen3 実行スクリプトの開始前ガードと合わせています。

## 実行方法

リポジトリ直下で実行します。

```bash
PYTHONPATH=src python -m vare rsi preflight \
  --repo-root . \
  --run-root . \
  --output outputs/rsi-preflight.json
```

成功時は終了コード `0`、実行要件が不足または未確認なら `2` です。`--model-path` を指定する場合も、固定したリポジトリ名とrevisionを含むHugging Face snapshot配置を使います。別モデルや任意のディレクトリは固定モデルとして通しません。任意の出力先に JSON を書き、ファイルは同一ディレクトリ内で置換して保存します。

## 判定の範囲

`runtime_status: ready` はローカル実行要件だけを意味します。タスクの新規性・割当、grader の独立性、学習データの有効性、方策更新、独立評価は判定しません。JSON でもそれぞれ `not_assessed` または `not_run` と記録します。したがって、この結果だけでエージェント実行や学習を開始してよいとは判断できません。対象の固定 protocol とタスク別の安全・評価ゲートを別途満たす必要があります。

## Integration with full-trajectory GRPO

The runtime preflight and strict v3 checkpoint proposal is reconciled with
the complete-action sampler-integrity candidate. The lag-zero guard,
opt-in native sampler score mode, verified group admission, and original
numerical artifacts are preserved. All current checkpoint callers in the
active scripts and the full-trajectory regression supply a base model ID.
Historical source snapshots and result artifacts remain unchanged.

The v3 loader intentionally rejects v1/v2 checkpoints and validates base
model identity, optimizer class/public configuration, file hashes,
adapter keys/shapes/dtypes, integer update counts and RNG metadata before
restoration. The ID is caller-provided; it is not proof of the base tensor
bytes. These checks do not authenticate an untrusted manifest, establish
arbitrary optimizer-state layout compatibility, or authorize training.
Use the exact historical code to inspect legacy checkpoint evidence;
do not relabel an old checkpoint as v3. No migration of historical
checkpoints is performed here. The added scalar-step regression reseals
a fractional optimizer step and requires refusal before model mutation.

## ResearchGym integration

The merged ResearchGym v8/v9 continuation source sets are preserved. The
RSI-only proposed v8 set is retained as `v8-rsi-preflight`; the combined
current source inventory is v10 (ResearchGym v9 plus `src/vare/rsi_cli.py`).
Both CLI routes remain available. Frozen artifacts are not rewritten.
