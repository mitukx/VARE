# Sprint XIII 実験報告 — オンラインActor、検証器ドリフト、SIGKILL復旧

**実施日:** 2026-10-29 JST
**対象:** VARE / Sprint XIII — online actor learning, imperfect verifier, long-horizon RL

## 結論

Qwen2.5-0.5B-Instructを使い、可変長の合成trajectory生成、実optimizer更新、更新後policyからの再生成、verifier v0/v1のドリフト、実SIGKILLと別PIDでの再開を接続した。凍結済み15ケースのうち13ケースは完了し、その13件を新しい独立監査器で全件再生した。26件の完了済みoptimizer commit、token logprob、gradient/Adam/RNG/checkpoint、trusted evaluation、5種の障害cutpointを検証した。3 seedすべてでD/Eの最終adapter、optimizer、RNGが完全一致した。

これは**合成route選択課題に対する小さなモデル連動systems実験**である。13完了caseすべてでtrusted評価はseedごとの初期値から変化せず、外部repository修復は0件である。Coding RLの能力改善、held-out改善、RSIによる自律的な研究選択は示していない。キャンペーンはメモリ圧迫ではなく、凍結した1,200秒wall-clock上限で15ケース中13ケースの時点で停止した。

## GitHubとmainline

`git fetch --all --prune`後に確認した開始時・終了時の`origin/main`はともに `40f7b38fe619217c0ebd534c4d89d42617859be8`。mainは保護されている。Sprint XIIIで既存PRを更新・close・mergeした件数は0。今回の実験コード・凍結protocol・結果・監査は[Draft PR #148](https://github.com/mitukx/VARE/pull/148)にまとめた。作成時headは`519995fc14253e68c7266c33ac43922524bbac89`、baseは上記main SHA。作成直後のlive open数は19（Draft 10）で、#148には独立review 0件、thread 0件、head check 19件が実行中だった。最初の18件のPR台帳は作成前のsnapshotであり、#148を含まない。Draft作成はmainline integrationとは数えず、独立reviewと必要なCIが揃うまでmerge対象にしない。

取得したopen PRは18件、Draftは9件。全18件に独立した`APPROVED` reviewはなかった。live snapshotの機械可読台帳は [`github-open-pr-audit-v1.json`](../results/sprint_xiii_20261029/github-open-pr-audit-v1.json)（SHA-256 `1dea493c026d15e24576bd7eba39795ec2eeea41674ca5b42037a18ea77404b0`）。check-run件数は同じhead上の再実行を含むGitHub記録件数である。

| PR | Draft | 現在のhead checks | 未解決thread | 判断に関わる状態 |
| --- | --- | ---: | ---: | --- |
| #146 | はい | 20/20 success | 0 | v9/v10測定は24 trajectory中20で停止した既存のNO-GO。v9 defaultを維持。 |
| #145 | いいえ | 19/19 success | 1 | family-specific契約v2。P2 threadでreviewer確認待ち、main baseは古いSHA。 |
| #142 | はい | 19/19 success | 0 | evidence PR。作者自身のreview以外の必須independent review待ち。 |
| #139 | はい | 20/20 success | 0 | #137に積まれたDraft。単独merge対象ではない。 |
| #137 | いいえ | 28/29、1 failure | 0 | `test (3.12)`が失敗。#139の親ブランチ。 |
| #136 | いいえ | 19/19 success | 0 | source-navigation stack。 |
| #135 | はい | 19/19 success | 0 | preflight/evidenceのみ。Draft。 |
| #133 | いいえ | 20/20 success | 2 | P2: path inventory予算と一時index build失敗からの回復。親stackも未解決。 |
| #131 | はい | 20/20 success | 0 | hash-bound edit/recoveryのDraft evidence。 |
| #126 | いいえ | 20/20 success | 0 | verifier-aware GRPOの履歴PR。独立承認なし。 |
| #122 | いいえ | 28/28 success | 0 | 7B teacher NO-GO記録。独立承認なし。 |
| #121 | いいえ | 19/19 success | 0 | Transformers source replay。独立承認なし。 |
| #120 | いいえ | 20/20 success | 2 | P1: grader completion forgeryと空evidence manifest。mergeabilityもfalse。 |
| #118 | はい | 20/20 success | 0 | structured edit libraryのDraft extraction。 |
| #117 | はい | 19/19 success | 4 | MLX-LM operator stack。未解決P1/P2と親PR依存。 |
| #114 | はい | 19/19 success | 0 | MLX-LM evidence preservation。#99の未解決trust境界を継承。 |
| #108 | はい | 19/19 success | 0 | dual-Python protocol Draft。 |
| #99 | いいえ | 20/20 success | 3 | P1/P2のgrader/source-integrity指摘が未解決。 |

green CIは独立承認やmerge適格性を意味しない。特に#120、#99、#133、#137の指摘をこのSprintで解決したとは報告しない。

## 仮説、固定条件、モデル

**仮説:** verifier v0からv1への不完全な更新下で、stale rewardの受け入れ、use-time rejection、trusted regradingの選択が、実際のQwen policy gradientとtrusted outcomeに影響し、fault-aware arm Eは同一seedのrecoveryなしarm Dと状態一致を維持できるか。

主実験protocol v4は生成前に固定した。SHA-256は `194433a9fbbfaae9d8968f5c72a1f4cd4c1ca019de3470c35e513edda8a6ed23`。source commitは `c82c5d9ae51ec4e68c56dae73aa3ec8c6844df8d`。task manifestは2 train route promptと4 disjoint synthetic evaluation promptを固定し、seedは707/808/909、生成数16/group、temperature 2.0、trajectory長2–5 action tokensとした。評価単位は4 evaluation promptであり、responseや試行数を独立taskとして扱わない。

- **モデル:** `Qwen/Qwen2.5-0.5B-Instruct`, Hugging Face revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- **weights:** `model.safetensors` SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。tokenizerを含むsnapshot内10 fileのhashもprotocolに固定。
- **backend/runtime:** CPU fp32、Python 3.12.12、PyTorch 2.9.1、Transformers 4.57.3。Metal/GPU使用なし。
- **学習対象:** pretrained baseはfreezeし、vocabulary-sized fp32 output logit biasの151,936 parameterのみを更新。
- **外部learner:** Recursive-Verification-Lag commit `c7e646b043cb56e5ea3c2623bb8a61e065451f72`。
- **resource caps:** 1,200秒campaign、child RSS 16 GiB、available RAM開始時4 GiB以上/hard floor2 GiB、swap増加上限512 MiB、free swap開始時512 MiB以上、同時child最大2。capは変更していない。

比較armはA同期current-verifier、B stale-v0 rewardのnegative control、C use-time stale group reject後のfresh replacement、D current-v1へのregrade、E Dと同じregradeにSIGKILL/recoveryを追加、の5群。seedごとの順序はprotocolで固定・counterbalance済み。

## 実行履歴と失敗の保存

protocolや監査器の不具合で失敗した記録も、再実行や上書きせず残した。

1. **protocol v1** (`bb377473…e3130dd`): `tasks`欠落でcalibration時に`KeyError`。model forward/generation前の無効化。
2. **run-001 / protocol v2** (`6bad616f…a72800b`): `sample_batch`後に`run_id`欠落でgroup保存前に停止。生成済み16 sampleのraw出力は保持できなかった。再実行はしていない。
3. **run-002 / protocol v3** (`8ecac1a1…3b80db3`): 新しいseed/promptで16出力を永続化し、v0採点まで進んだが、learnerのsource-root key欠落でoptimizer前に停止。62 generated token。独立adjudicationは16件中2件trusted-positive、proxy/trusted disagreement 13/16。
4. **run-003 / protocol v4**: 1,200秒で停止。13件の`case-result.json`、`seed-909-C`のpartial case、未完了の`seed-909-B`と`seed-909-C`を記録。完了判定に達しないものを成功に数えていない。
5. **監査v1**は初回audit resource-key不一致、**v2/v3**はacknowledgement JSONをdataとして誤読して中断。各プロトコルとfailure recordは残してある。
6. **監査v4**は全13完了caseを再生したが、`seed-909-E`のpre-publication candidateを未保持と報告。調査すると、SIGKILL前のattempt-1 candidate file/manifestは実在し、再試行attempt-2が決定論的に同じbytes/hashを生成してcommitしたため、v4 auditorの「orphan hashは後続commit hashと異なる必要がある」というpredicateが誤っていた。
7. historical v4 receiptは変更していない。hash-bound regressionを追加し、監査protocol v5（SHA-256 `eeafde7df897414643e66ba769e31518b662bbac0f02134e739e2c5f527cef7f`）を**監査再実行前に別版として固定**。v5はfaulted attempt record、candidate artifact、manifest、hashを個別照合する。

## 独立監査と実験結果

監査v5の結果は`PASS_COMPLETED_SUBSET`、13/13 completed cases、findings 0。キャンペーン全体は13/15なので`campaign_complete=false`のままである。監査はcampaign protocol/source/model/RVL identity、保存tokenごとのbehavior logprob、proxy/trusted reward、independent GRPO gradientとAdamW/RNG/checkpoint replay、journal/admission/freshness、trusted evaluation、5 fault cutpoint、resource recordを再計算した。replayed gradientのmax absolute errorは0。

### Trusted evaluation

各seedでbaselineも更新後armも4 synthetic evaluation prompt × 8 generation = 32 samples。すべての完了armでseed内baselineに対する各task差分が0.0だった。

| Seed | Baseline trusted positive | 完了arm | 更新後 trusted positive | mean reward差 |
| ---: | ---: | --- | ---: | ---: |
| 707 | 3/32 (0.09375) | A/B/C/D/E | 全arm 3/32 | 各task 0.0 |
| 808 | 4/32 (0.12500) | A/B/C/D/E | 全arm 4/32 | 各task 0.0 |
| 909 | 1/32 (0.03125) | A/D/E | 各arm 1/32 | 各task 0.0 |

seed 909のB/Cはwall-cap前に完了していないため比較に入れていない。評価結果は能力改善を示さない。3 seed・4 synthetic promptの少数コホートなので有意差や一般化範囲の主張をしていない。

### Verifierの誤りとgradient影響

trusted evaluatorから隔離した事前固定のenumerationでは8,160 action stringについて、v0はfalse positive 3,186 (39.04%)、false negative 126 (1.54%)、trustedとの差40.59%、v1はfalse positive 78 (0.96%)、false negative 0。v0/v1 disagreementは39.63%。実際に保存したtraining responsesではv0 368件中88件がfalse positive、v1 176件中0件がfalse positiveだった。

seed 707/808ではBとDの2nd-update groupのtoken IDsが一致し、score hashは異なる。保存されたpost-clipの5 support-coordinate gradient projectionのcosineはそれぞれ0.97264、0.97201。これは限定された座標投影であり、full-vocabulary gradientのcosineではない。rewardの再採点が更新信号を変えた診断証拠だが、trusted task successは変わらなかった。

### オンライン生成、更新、障害回復

- 13 completed casesで26 effective commits。さらに`seed-909-C` incomplete directoryに1 commit eventがあるが、complete caseとして監査・集計していない。journal全体は27 commit events、29 optimizer-applied events（うち2計算はSIGKILL後の再計算）。
- 13 completed casesすべてでpolicy v2 checkpointからfresh-policy-probeを実生成。監査したprobe distributionと直前groupのfirst-step distributionのTVは0.000746–0.002875、KLは約2.38e-6–2.13e-5。
- 完了caseのtraining/replacement/probe groupは552 generations、2,169 generated tokens。trajectoryは2–5 action tokensの可変長。token-level logprobを独立再生した。これは合成route sequenceであってcoding-agent tool trajectoryではない。
- A/B/C/D/Eの13完了caseのqueue-inclusive update latency 26点はmedian 26.73秒、p95 37.48秒。単一armのthroughput優位性は主張しない。
- 3 seedのD/E比較でadapter、Adam optimizer、RNG stateはすべてbitwise equal。D/Eそれぞれのtrusted outcomeも一致。
- 5 fault cutpointすべてで実SIGKILLを観測し、5回すべて異なるPIDで再起動。記録されたkill-to-child-respawn時間は5.1–6.4 msであり、これはprocess respawn gapであってmodel reloadを含むend-to-end recovery latencyではない。
- completed casesのchild peak RSSは9,069,985,792 bytes、host available RAM minimumは8,962,087,680 bytes、swap growth 0。independent auditは205.25秒、peak RSS 3,796,418,560 bytes、swap growth 0。campaign停止理由はwall-clock capで、OOM/resource-pressure abortではない。

## Claim gate

| 領域 | 状態 | 判定 |
| --- | --- | --- |
| Schema/trajectory/actor refresh | 完了caseでRUN | 実Qwenが可変長sequenceを生成し、更新policy v2からfresh probeを生成。全campaign完走は未達。 |
| Online GRPO更新 | RUN、限定的 | 26 audited effective commits。trusted synthetic outcomeは改善なし。Coding RLとは呼ばない。 |
| Verifier drift/admission/regrading | PARTIAL | Stale/proxy誤りと更新信号差を測定。use-time controlsとD/E整合はaudit pass。15/15ケース比較ではない。 |
| Long-horizon coding task | NOT RUN | 最大5 action tokensの合成sequenceであり、長期coding-agent能力を示さない。 |
| 外部Coding task / verified repair | NO-GO | このSprintの新規外部task 0、独立修復0。 |
| Coding SFT/GRPO | NO-GO | 適格なcoding reward trajectoryがなく、held-out gateも未充足。 |
| Held-out improvement | NO-GO | 既存source/grader-certified count 8/15、model exposure eligibility UNKNOWN。 |
| RSI adaptive refresh | NOT RUN | hard-coded schedule外のadaptive research-selection実験なし。 |
| mainline merge | NOT RUN | independent approvalがなく、branch protectionを維持。 |

## 再現と証拠

Campaignの13完了case、partial case、process logs、raw model outputs、v4/v5監査receiptは[証拠アーカイブ](../results/sprint_xiii_20261029/sprint-xiii-run-003-evidence-v1.tar.gz)にlossless収録した。archive SHA-256 `91b8985859ce4c9293f5a720d02b3be2d5e02dc99482bac4602b48f5acd87a65`。アーカイブ内の`run-003/sha256-manifest-v1.json`は622個のraw fileを列挙し、manifest SHA-256は`74bcf347fb636d7224f4c60ff08a7aa6da575b5d9f275a639bbc8f51539f4c9a`。展開時に622件すべて一致することを確認した。v4失敗auditとv5成功auditの両方を保存してある。

ローカル再現ではarchiveを展開し、凍結したmodel/RVL/source snapshotを用いる。

```bash
tar -xzf results/sprint_xiii_20261029/sprint-xiii-run-003-evidence-v1.tar.gz \
  -C results/sprint_xiii_20261029
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /tmp/vare-sprint-xi-cpu/bin/python \
  scripts/sprint_xiii/audit_completed_online_actor_subset_v5.py \
  --protocol protocols/sprint_xiii_online_actor_v4.lock.json \
  --output results/sprint_xiii_20261029/run-003/independent-audit-v5-reproduction.json
```

The 1,200-second model campaign is not rerun by this audit command. Audit-v5 code SHA-256は[`audit-v5-protocol.json`](../results/sprint_xiii_20261029/audit-v5-protocol.json)に記録した。model weightsはGitに含めていない。protocolのlocal pathを別hostで使う場合はsame-byte assetsを展開し、固定hashを照合する必要がある。

## 次の研究判断

最も有用な次段階は、同じ消費済みprompt/seedを再利用せず、別task manifestと新seedでverifier比較をprospectiveに再登録し、停止時間だけを実測に合わせて設計すること。memory/swap capは維持する。全caseを完了できる予算を確保し、fresh actor behaviorにtrusted reward差が出るかを先に測る。実task修復を主張する前に、Development grader資格とsource/exposure splitを独立に完成させる。

このSprintで正当に追加できた主張は、**限られた小規模CPU実験において、実pretrained modelの更新済みactorが新trajectoryを作り、不完全verifierのstale reward/regradingによる更新差を観察し、5種類のSIGKILL境界後に3 seed中の対応D/E状態を独立に同一と再計算できた**ことまでである。task能力向上、coding repair、held-out一般化、recursive research improvementは示していない。
