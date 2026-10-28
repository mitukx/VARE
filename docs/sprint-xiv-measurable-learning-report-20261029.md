# Sprint XIV — 学習診断、Coding Agent実測、main統合

**記録日:** 2026-10-29 JST
**対象:** VARE / mainline `mitukx/VARE`
**研究記録:** negative/partial evidenceを含む。ここで報告する実装修正をモデル能力の改善と同一視しない。

## 結果要約

- main `40f7b38fe619217c0ebd534c4d89d42617859be8` から、Development課題ごとのpublic-test有無とtool allowlistを固定する [PR #145](https://github.com/mitukx/VARE/pull/145) をsquash mergeした。merge commitは `b1e5882cbf89a8f925bf6e5f0a2f288f2d4c8a02`。そのmainのCI run [40782648279](https://github.com/mitukx/VARE/actions/runs/40782648279) は9 jobsすべてsuccess。
- Qwen2.5-Coder-7Bで新規Development family #49752を1回実行。source readとmultiline source searchを含む3つの連続tool callはschema-validだったが、その後2回toolを呼ばず終了した。edit 0、public test未実行、独立graderは変更なしのbaselineをreject。修復成功は0/4 family。
- Qwen2.5-0.5BのSprint XIII既存policyを正確に列挙評価した結果、78 policy/task比較のうちtrusted expected rewardは77件で低下、平均差は −0.00085935。main-based integration branchから独立再実行し、主要metric JSONがhistorical outputと完全一致した。これは消費済みsynthetic cohortのretrospective分析であり、新しい学習やCoding RL結果ではない。
- SFT/GRPOは0 update、held-out改善評価も未実施。held-out family certificationの最新記録は8/15で、学習gateはNO-GO。
- PR #145をmainへ統合し、mainlineでのプロトコル成果を一つ増やした。残る18 PRのうち10件はDraft。さらにmain-basedの焦点を絞った[PR #149](https://github.com/mitukx/VARE/pull/149)を作成し、レビュー待ちでopen。作成直後のheadは `2a4ef1de33d2ae9839ab001f88febd361e11e471`、baseはmain `b1e5882cbf89a8f925bf6e5f0a2f288f2d4c8a02`。この報告更新を同PRへ追加するためheadは後続commitで更新される。required reviewがないためmergeしていない。

## 1. 実行した研究仮説と境界

### 学習診断

問いは「optimizerを動かせるか」ではなく、更新がtrusted objectiveをどう変えたかである。Sprint XIIIで消費済みの記録を、frozen protocol `sprint_xiv_exact_policy_expectation_v1` に従って再計算した。モデルは `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`、CPU fp32、weights SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。historical artifact 622件から13 candidate policies、6 promptsを使い、各promptの有限出力空間1,360系列を列挙した。新規生成tokenもoptimizer更新もない。

| 指標 | 観測 |
|---|---:|
| paired policy/task comparisons | 78 |
| trusted expected reward差 | 平均 −0.00085935、77/78が負、1/78が正、範囲 −0.00268218〜+0.00013668 |
| exact task success probability差 | 平均 −0.00087559、65/78が負、13/78が正 |
| trajectory total variation | 平均 0.0051567、範囲 0.0036174〜0.0058238 |
| KL(candidate‖baseline) | 平均 8.2177e−5 nats、範囲 4.2685e−5〜1.0186e−4 |
| first-step probability calibration error | 最大 2.03e−6 |
| new tokens / new optimizer updates | 0 / 0 |

78行はseedと6 promptの反復を含み、独立task familyの78例ではない。ここから信頼区間や汎化効果は計算しない。全policyの変化が小さく、trusted expected rewardはほぼ一貫して悪化していることは示せるが、gradient方向、Adam/clipping、reward proxy、global-bias adapterの表現力、explorationのどれが原因かを分離しない。context-dependent adapter、trusted/imperfect verifierの因果比較は実行していない。したがって原因は「小さなpolicy shiftと整合する」が、単一の根本原因を確定したわけではない。

関連artifact: [再計算レポート](../results/sprint_xiv_evidence_v1/learning_diagnosis/report.md)、[exact output](../results/sprint_xiv_evidence_v1/learning_diagnosis/exact-policy-analysis.json)、[frozen protocol](../protocols/sprint_xiv_exact_policy_expectation_v1.json)。

### 外部Development task qualification

以下4 familyはApache-2.0のTransformers過去修正から作成した。source SHA、候補baseline、public/hidden verifierと制御結果は [task evidence manifest](../results/sprint_xiv_evidence_v1/sha256-manifest.json) に収録した。各familyでbaseline FAIL、known fix PASS、wrong mutant FAILを確認し、hidden controlsでは候補による静的stdout偽装もrejectする。pretraining exposureは全てUNKNOWN。これらはDevelopment課題であり、blind held-out評価ではない。

| Family | Baseline → fix commit | Source SHA baseline → fixed | 適格性と露出 |
|---|---|---|---|
| Transformers PR #49900 — NumPy label dtype | `50a952d06c8f` → `5922731af57e` (merge `1d20b1ae431a`) | `6f458e446656` → `7f2eb94b8871` | baseline/fix/wrong controls PASS; Development exposure済み |
| Transformers PR #50044 — Janus string prompt | `a1a59ea46ee8` → `2d0bb18d7cc9` (merge `c27795950091`) | `d87652762ba0` → `fe1caf4a8371` | controls PASS; Development exposure済み; model runはworkspace-integrityで無効化 |
| Transformers PR #50021 — TimesFM fp16 pad value | `6e06886cc3c9` → `0cca4b0e3685` (merge `24f3b85a2d95`) | `7b625e923ac8` → `b1b9a385176b` | baseline/fix/wrong/static-spoof controls PASS; Development exposure済み |
| Transformers PR #49752 — MiniCPM-V stacked visual units | `6f2e80bdad56` → `482e80ae5910` (merge `bc0e71ffef22`) | `c0479fc2f6a1` → `19f2f1b936c9` | baseline/fix/wrong/static-spoof controls PASS; Development exposure済み |

#49752の最初のpublic probeはgrader harnessへ `group_videos_by_shape` callbackを注入しておらずNameErrorになった。失敗receiptを保持し、修正済みのprobe v2でbaseline/fix/wrong controlsを再確認した。初回失敗は資格認定に使っていない。

### 実モデルattempts

同じcached checkpointとresource capsを使ったが、過去3 taskは別task/seed/prompt条件であり、matched controller comparisonではない。ここで比較優位や効果量は主張しない。

| Attempt / task | Model output / tool behavior | Edit・test・grader | 計測 |
|---|---|---|---|
| 101 / #49900 | 4生成、3 call (`list_files`, `read_file`, `read_file`)、70 tokens。sourceを再読してcontext budget到達。 | editなし、public test未実行。independent graderは変更なしbaselineをreject。 | 155.5s agent / 160.7s guard; process RSS 4.43 GiB; MLX peak 5.51 GiB; swap growth 0。 |
| 102 / #50044 | model process起動後にworkspace copyでtracked symlinkがregular file化し、canonical source checkが拒否。raw transcript未保存。 | invalid infrastructure attemptとして除外。edit/test/graderなし。 | 113.8s; process RSS 4.40 GiB; swap growth 0。resource abortではない。 |
| 103 / #50021 | 6生成、4 call (`list_files`, `read_file`, `search_text`, `read_file`)、148 tokens。旧searchでmulti-line queryを見つけられず、再読が発生。 | editなし、public test未実行、grader未実行。 | 197.0s agent / 201.8s guard; process RSS 4.37 GiB; MLX peak 5.59 GiB; swap growth 0。 |
| 104 / #49752 | 5生成、3 call (`list_files`, `read_file`, `search_text`)、83 tokens。searchは3行の一致を返した。残り2生成はtool callなし。 | editなし、public test未実行。nonce-bound independent graderはunchanged baselineをreject。 | 93.8s agent / 96.9s guard; process RSS 4.39 GiB; MLX peak 5.12 GiB; swap growth 0。 |

checkpointは `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`、MLX/Apple Metal、revision `019cc73c45c770444708a6dd8690c66243cc5c80`、weight file SHA-256 `56a3d94706833f753e6c6b47ea57af8ef638cd8cc1d74eca1142ba640b26060e`。Python 3.12.12、MLX/MLX-LM 0.32.2。各attemptは独立したguard下で逐次実行した。

Frozen caps: max process RSS 10 GiB、MLX peak 8 GiB、swap growth 32 MiB、最低available memory 2 GiB、free swap 512 MiB、disk 16 GiB、wall 900 s、context 32,768 tokens、最大12 turn、512 token/turn、4096 tokens/attempt。capsは上げていない。model process 4回のうち3 trajectory保存済み、1 infrastructure-invalidatedでraw outputなし。保存済みtrajectory合計301 generated tokens、10 accepted tool calls/15 generated responses。3/3 #49752 tool callsは正しいtool nameとargumentsでparserに受理された。model outputのfence/ChatML外側表現にはrunnerのstrict wrapper normalizerを使い、argumentsを書き換えてはいない。これはgrammar-constrained generationの実証ではない。

### Controller介入と原因

#50021で観測した原因は二つ: searchが行内のliteral一致だけで複数行式を見つけられず、sourceを既に読んでもno-tool reminderが再読を促した。task固有文字列を埋め込まない一つの介入を新規#49752 Development taskで試した。

- `search_text` はqueryの非空白文字列を保ち、空白/改行を柔軟に照合し、bounded line spanを返す。
- source read後にtool callがない場合、全文再読ではなく短いsearch/minimal edit/checkを促す。
- task source hash/clean Git treeをmodel load前に照合する。
- candidate diff validation失敗時もraw trajectoryを捨てず、エラー付きrecordを保存する。

新しいtaskでは検索と継続制御の両方が作動し、不要な全文再読は起きなかった。一方、search後のmodel出力は不完全なChatML start markerだけとなり、二度目のreminder後もtoolを呼ばなかった。これはsource localization失敗ではなく、今回のtraceではaction selection / no-tool generationで止まった。介入が修復率を上げる仮説は支持されず、search callの到達性のみ確認された。

Regression suiteは78 tests passed。対象はagent tool formatting/edit format/container isolation/workspace binding/Sprint XIV exact-policy analysisの各suite。`ruff check`、`py_compile`、`git diff --check`も成功。全VARE test suiteを実行したという意味ではない。

### Exact policy replayとmainline依存境界

同じ圧縮run archive（SHA-256 `91b8985859ce4c9293f5a720d02b3be2d5e02dc99482bac4602b48f5acd87a65`）、campaign lock（SHA-256 `194433a9fbbfaae9d8968f5c72a1f4cd4c1ca019de3470c35e513edda8a6ed23`）とanalysis protocolでmain-based integration branch上から解析を再実行した。78件のtrusted reward差、task success差、全policy exact metric、first-step calibrationがhistorical outputと完全一致。再実行は195.4秒、peak RSS 3.17 GiB、generated tokens 0、optimizer updates 0である。全出力・preflight・hash manifestは[再実行記録](../results/sprint_xiv_evidence_v1/learning_diagnosis/reproduction-v1/report.md)に保存した。

もとのanalysis helperはunmerged #148 runtime `scripts/sprint_xiii/online_actor.py` に依存していたため、mainline側では、そのhistorical Git blob `f3b27930b49e43a92810f35b4f9f8d2fe195589c`（file SHA-256 `0c8899ac9b2177df5593b3f608abcfdda287ef916f43eee119dc94e9a1f63c3b`）から純関数 `prompt_text()` と `verifier_score()` のみを出所コメント付きで移植した。historical helperとmainline helperを比較し、1,365文字列（長さ0〜5のABCD列挙）×2 verifier versionすべてでscoreが一致し、prompt文字列と凍結prompt-token IDsも一致した。これは計算経路の再現性確認であり、#148のactor implementationを統合したことや、新しい学習効果を示すものではない。

## 2. Long-Horizon RL、Coding RL、RSI、training gate

- **実model long-horizon:** 今回は未実施。最長の保存済みattemptは6 model turns、継続再開・fault injection・effective commitは0。model-coupled recovery/async throughputの結果として扱わない。
- **既存synthetic systems evidence:** READMEの504 logical rounds、513 optimizer attempts、18/18 SIGKILL recoveryは既存のsynthetic-control evidenceのまま。今回再実行・更新していない。
- **Coding SFT/GRPO:** NO-GO。verified positive coding trajectory 0、eligible trusted reward-positive coding group 0。adapter updateを実行していない。GRPO optimizer activityだけを示すrunもない。
- **held-out improvement:** NO-GO。latest repository recordは8/15 certified held-out familyで、15-family gate未達。今回の4課題は全てDevelopment exposure済みで、再利用不可。
- **RSI research automation:** NOT RUN。failure taxonomyはhuman-chosen diagnosisであり、task-selection policyが次の課題や結果を自律的に変えた比較はない。RSI成果とは呼ばない。

次に行う価値が高い実験は、未使用の新Development familyで、search後のno-tool stateからactionを一つ選ばせるcontroller-policy ablationをprospectiveに固定すること。成功指標は実際に適用されたpatchと独立grader合格であり、tool-call数ではない。現taskを再利用しない。並行してtask poolが15-family held-out gateへ達するまではmodel learningを開始しない。

## 3. GitHub auditと統合

main開始時は `40f7b38fe619217c0ebd534c4d89d42617859be8`。#145はexact head `1a99df2861b79bedc29ca400a35505bceb47c686`、baseは開始main、19/19 checks成功。GraphQL unresolved review threadは0、branch protectionのrequired approving review countは0。最新レビューはautomated/owner COMMENTEDであり、どれも独立approvalとして数えていない。main保護やrequired CIを無効化せずにmergeし、post-merge main CI run 40782648279は9/9 jobs success。merge commit `b1e5882cbf89a8f925bf6e5f0a2f288f2d4c8a02`。

#145でmainlineに入った内容: 6つのDevelopment familyごとのpublic-test availability/tool allowlist contract v2とそのtests。これはmodel capability resultではない。

### PR台帳

開始時は19 open / 10 Draft。#145 merge後は18 open / 10 Draft、1 merged、0 closed-without-merge。その後#149を作成したため現在19 open / 10 Draft、期間中1 merged / 0 close-without-merge / 1 created / 0 existing-PR updated。#149はready for reviewだが未レビューで未merge。full exact SHA、base/head、CI, review threads, changed file inventory, dependencies, rationaleは [machine-readable PR ledger](../results/sprint_xiv_github_audit/pr-ledger-sprint-xiv.json)、依存図は [dependency graph](../results/sprint_xiv_github_audit/dependency-graph-sprint-xiv.mmd)。ledgerの既存18件は#149作成前の全件監査snapshot、#149は後述の追加行である。

| PR / title | State | Head / base prefix (merge state) | CI | 未解決レビュー | 主分類と次の作業 |
|---|---|---|---|---|---|
| [#99](https://github.com/mitukx/VARE/pull/99) Calibrate isolated MLX-LM 1971 Development task | Ready | 762a5a6559 / f3ab90a36e (BEHIND) | 20/20 | P1, P1, P2 | FIX_THEN_MERGE — Repair the cache/scale and preflight paths, then rerun adversarial controls. |
| [#108](https://github.com/mitukx/VARE/pull/108) Add dual-Python HF Coding Development protocol v2 | Draft | 710dbc0d28 / ab8d7c0b55 (CLEAN) | 19/19 | なし | EVIDENCE_ARCHIVE — Keep evidence immutable; extract any still-needed small implementation onto main. |
| [#114](https://github.com/mitukx/VARE/pull/114) Preserve attested MLX-LM calibration evidence on publication failure | Draft | 8257a4dfe2 / 0d843a3e2e (CLEAN) | 19/19 | なし | BLOCKED — Retain the evidence; rebase/extract only after #99 audit. |
| [#117](https://github.com/mitukx/VARE/pull/117) Isolate MLX-LM source snapshots before trusted export | Draft | 6709aed996 / 8257a4dfe2 (CLEAN) | 19/19 | P2, P2, P1 | FIX_THEN_MERGE — Fix each finding with independent regressions, then retarget after parent review. |
| [#118](https://github.com/mitukx/VARE/pull/118) Extract hash-checked structured edit library without changing frozen agent runtime | Draft | 552d6f6118 / aef27e3b7b (CLEAN) | 20/20 | なし | SPLIT_OR_CHERRY_PICK — Compare APIs with main, then extract only nonduplicated library code and tests. |
| [#120](https://github.com/mitukx/VARE/pull/120) Record real-model coding-agent feasibility and tool-boundary fixes | Ready | 8aacae54d0 / f3ab90a36e (DIRTY) | 20/20 | 2件 | FIX_THEN_MERGE — Repair trusted parent completion and reject empty evidence; calibrate baseline/fix/mutant before extraction. |
| [#121](https://github.com/mitukx/VARE/pull/121) Add independent source replay for Transformers fusion registration | Ready | a937f4ad6e / f3ab90a36e (BEHIND) | 19/19 | なし | BLOCKED — Rebase onto current main and validate replay invariants before considering merge. |
| [#122](https://github.com/mitukx/VARE/pull/122) Close installed-repair leakage and record the 7B teacher NO-GO | Ready | 37b3151981 / f3ab90a36e (BEHIND) | 28/28 | なし | BLOCKED — Rebase and rerun current-main CI; preserve its negative evidence. |
| [#126](https://github.com/mitukx/VARE/pull/126) Measure verifier-version-aware GRPO under drift and crashes | Ready | c728b9f2ec / a476feeed7 (CLEAN) | 20/20 | なし | EVIDENCE_ARCHIVE — Preserve raw evidence; extract a narrowly justified general mechanism only if absent from main. |
| [#131](https://github.com/mitukx/VARE/pull/131) Measure deterministic edit replay and recovery; pin E2E inventories | Draft | 139de53819 / 071352981a (CLEAN) | 20/20 | なし | EVIDENCE_ARCHIVE — Preserve source inventory and exact-head receipts; do not merge as a broad stack. |
| [#133](https://github.com/mitukx/VARE/pull/133) Add bounded source navigation for repair agents | Ready | e8ab4705b8 / da4bc58650 (CLEAN) | 20/20 | 2件 | FIX_THEN_MERGE — Add bounded-output and retry regressions, fix both, then rerun exact-head CI. |
| [#135](https://github.com/mitukx/VARE/pull/135) Add process-creation boundary preflight for 47468 v4 | Draft | b258c69fc0 / d8636def09 (CLEAN) | 19/19 | なし | SPLIT_OR_CHERRY_PICK — Verify the code is still needed, then extract with its tests if nonduplicated. |
| [#136](https://github.com/mitukx/VARE/pull/136) Harden source navigation syntax checks | Ready | eaf3fd58dd / a65d59e570 (CLEAN) | 19/19 | なし | SPLIT_OR_CHERRY_PICK — Extract only the syntax-boundary fix after #133 defects are resolved. |
| [#137](https://github.com/mitukx/VARE/pull/137) Fix source-navigation review findings | Ready | 6d95bd9196 / eaf3fd58dd (CLEAN) | 20/20 | なし | FIX_THEN_MERGE — Do not merge the stacked branch yet; resolve the upstream #133 P2s and #136/#137 dependency ordering, then rerun current-base CI. |
| [#139](https://github.com/mitukx/VARE/pull/139) Fix navigation regressions and measure fresh Development agent failures | Draft | df0a360ead / 6d95bd9196 (CLEAN) | 20/20 | なし | SPLIT_OR_CHERRY_PICK — Extract the minimal runtime/test patch; archive all run records append-only. |
| [#142](https://github.com/mitukx/VARE/pull/142) Publish verified external repair campaign evidence | Draft | 91d3d19a4c / f3ab90a36e (BEHIND) | 19/19 | なし | EVIDENCE_ARCHIVE — Preserve evidence; split task assets from public report and obtain human review before any integration. |
| [#146](https://github.com/mitukx/VARE/pull/146) v9/v10 multi-turn shared-prefix ABBA measurement (partial run-001) | Draft | 027e16aebb / df0a360ead (CLEAN) | 20/20 | なし | EVIDENCE_ARCHIVE — Retain raw outcomes; mark incomplete and preserve dependency on #139 evidence. |
| [#148](https://github.com/mitukx/VARE/pull/148) Sprint XIII: online actor learning and imperfect verifier evidence | Draft | 493740b39a / 40f7b38fe6 (BEHIND) | 19/19 | なし | SPLIT_OR_CHERRY_PICK — Extract stable code/tests and separate the report/data; keep the 13/15 negative run unchanged. |
| [#145](https://github.com/mitukx/VARE/pull/145) | MERGED | 1a99df2861 / 40f7b38fe6 | 19/19 success | 0 open threads | mainへ統合 |
| [#149](https://github.com/mitukx/VARE/pull/149) Record Sprint XIV verified agent research | OPEN / Ready | initial head `2a4ef1de33d2` / base `b1e5882cbf89` (report follow-up commit updates head) | live exact-head status in PR | independent review not yet received | MERGE_CANDIDATE — Review controller integrity changes and research evidence; do not merge before required review/CI. |

重要な阻害点は、#99の2 P1+P2、#117の4 source-isolation thread、#120の2 P1 evaluator/evidence issuesとconflict、#133の2 P2 source-index issues。#137の最新exact-head CLI checksは20/20 successだが、#133/#136からのstackをレビューせずにmergeする根拠にならない。#142は明示的なsplit/human-review blocker。#148はmainよりbehindでDraft、143 files/約32k additionsを含む。supersededと断定できるPRは見つけず、履歴破棄を伴うcloseは行っていない。

**Sprint内のPR操作:** #145 merged 1、既存PR update 0、without-merge close 0、focused PR #149 created 1。#149作成後のopen countは19/Draft10。#149は独立レビューがないため未merge。

## 4. Reproduction and evidence

- Exact policy replay: `tmp=$(mktemp -d); tar -xzf results/sprint_xiii_20261029/sprint-xiii-run-003-evidence-v1.tar.gz -C "$tmp"; python scripts/sprint_xiv/exact_policy_expectation.py --protocol protocols/sprint_xiv_exact_policy_expectation_v1.json --run-root "$tmp/run-003" --output results/sprint_xiv_exact_policy_expectation_replay_v1`。frozen campaign lockも保持。cached 0.5B weightsが利用可能なCPU環境で実行する。consumed artifactの再計算で、新規model generationではない。
- Narrow local checks: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_run_local_agent_tool_format.py tests/test_agent_edit_format.py tests/test_container_isolation.py tests/test_coding_agent_workspace_binding.py tests/test_sprint_xiv_exact_policy_expectation.py`
- Evidence archive: `results/sprint_xiv_evidence_v1/`。207 artifact entriesのSHA-256 manifestを検証し、mismatch 0。original raw evidenceはappend-onlyで、新しい4つのexact-replay出力を追加した。各attemptのcomplete raw `agent-record.json`, guard receipt, task controls, model preflight, hidden grader receipt, stderr/stdout、失敗したpreflight/classificationも保持する。大きなcandidate workspaces/model weightsはgitへ入れていない。
- Task/protocol identitiesとmodel invocation commandはprotocolとguard receiptに記録。4 consumed Development familyは再実行しない。

## 5. 最終判断

| 領域 | 状態 | 今回示したこと |
|---|---|---|
| Schema-valid tool interface | PARTIAL | 3連続正しいtool callを実modelで受理。grammar constraintではなくagent能力でもない。 |
| External coding task qualification | RUN | 4 source-fix familyでbaseline/fix/wrong controlsを確認。全てDevelopment露出済み。 |
| Independently verified code repairs | NO-GO | 0/4 family。成功trajectoryはない。 |
| Actual SFT / GRPO | NOT_RUN / NO-GO | verified positive exampleなし。optimizer update 0。 |
| Held-out model improvement | NOT_RUN / NO-GO | 8/15 latest certification record、gate未達。 |
| Model-coupled async / recovery | NOT_RUN | durable continuation/fault recoveryを今回測っていない。 |
| RSI research selection | NOT_RUN | adaptive task-selection comparisonなし。 |
| Mainline integration | RUN | #145 merged; merge `b1e5882`; post-merge CI success. |
| PR debt | PARTIAL | PR #145を1件統合、PR #149をmain-basedレビュー用に作成。現在19 open / 10 Draft。既存の重大findingはledger化。 |
| Next reviewable engineering result | PARTIAL | progress-aware source navigation + source integrity fixesをmain-based PRへ提出予定。 |

### 研究上の結論

このsprintの最も強いpositive resultは、7Bの実推論でschema-validな複数tool callが連続し、multiline searchが実ソースから結果を返したこと、そして当該attemptが安全な資源範囲内で完了したこと。ただしモデルはpatchを提出せず、独立graderはbaselineをrejectした。正当に示せるCoding Agent repair capabilityは今回も0件である。

最も重要なnegative resultは、search/localizationの欠陥を直しても、モデルが検索後にtool actionを生成しないこと。これは次の研究投資先をsearch qualityからpost-search action selection / tool-call continuationへ移す。学習側ではexact finite-policy evaluationで小さい分布変化とtrusted rewardの負差が確認されたが、更新機構やverifierのどれが律速かは未同定である。

したがって、今回の成果はxAI向けに「verified Coding RL improvement」「long-horizon learned agent」「RSI」を示すものではない。実測できたのは、外部task verifierを持つ7B agentの部分的なtool-use、再現可能な負のtask outcome、消費済みオンライン更新の正確なpolicy-level負結果、#145のmainline protocol integration、そしてsource-integrity/search regressionsの具体的修正である。次のcapability claimは、未使用taskで実際に適用されたpatchが独立graderを通るまで保留する。
