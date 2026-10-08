# CPU generated code-repair feasibility v1

## Result

The frozen base-only screen was a **non-pass**. Qwen2.5-0.5B-Instruct completed all 32 generated episodes, but it solved **0/32**. Success was predeclared to require a correct hidden-test result, an accepted source edit, at least one visible-test run, and an explicit `finish` call. None of the episodes reached an accepted edit, visible test, or finish.

The 32 episodes contain eight bug templates with four generated variants apiece. All four task families and all eight templates had zero successes. The aggregate rate is descriptive for this pilot; the repeated variants are not 32 independent task mechanisms.

| Measure | Observed | Frozen condition |
| --- | ---: | ---: |
| Hidden-test episode success | 0/32 | 8–26/32 |
| Success by family | 0/8 each | at least 2 in each |
| Success by template | 0/4 each | at least 1 in each |
| Tool calls | 0/69 schema-valid and authorized | at least 90% |
| Unsafe or unauthorized attempts | 60 | 0 |
| Episodes with accepted edit and visible test | 0/32 | at least 24 |
| Episodes with explicit finish | 0/32 | at least 24 |
| Complete, unique episode records | 32/32 | 32/32 |
| Wall time | 551.7 s | at most 7,200 s |
| Peak RSS | 3,430,170,624 bytes (3.20 GiB) | at most 6 GiB |
| Generated output | 6,707 tokens at 12.29 tokens/s | descriptive |

The tool trace explains the failure: 56 parsed `edit_file` calls used absolute paths outside the declared allowlist, four calls named non-existent tools, six calls had schema-mismatched arguments, and three outputs contained malformed JSON. The model did not first list or read the files, and the reminders did not recover its behavior. The runner denied all edits. These are failures of this frozen model/tool/task setup; they do not estimate coding ability across other models or interfaces.

## Protocol and audit

The run used the already-cached model revision `7ae557604adf67be50417f59c2c2f167def9a775`, Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, four CPU threads, offline model loading, and no model update, GPU, paid API, or external network request. Protocol SHA-256: `cb039334b64b8965d5c39987032daeb35cd807698d6879d78bd2bba62641dd0e`.

The independent auditor regenerated the task pack without importing the runner/generator/grader, re-evaluated candidate source with a separate stdlib AST interpreter, replayed each accepted edit and visible test, and reconstructed all metrics and gates. It returned `audited`; the frozen decision remains a non-pass. It did not repeat model inference. The retained bundle is [run-1](../results/cpu-code-repair-feasibility-v1/run-1/), with raw trajectories, source snapshots, task pack, summary, manifest, and `audit.json`.

Transformers emitted warnings that sampling-only defaults in the pinned generation configuration are ignored with greedy decoding and that no attention mask was supplied. The screen used one unpadded input at a time with a single prompt length; the run and effective generation configuration are retained so these conditions remain visible. No decoding or prompt change was made after observing results.

## Decision

Retire this model/task/tool pairing. Do not lower the thresholds, reuse the pilot, or turn these outcomes into an RL/SFT comparison. The resource gate passed, but the base did not provide successful trajectories and the action-safety/tool gates failed, so the proposed learner-cost smoke has no useful basis.

The next portfolio direction should lean on evidence this repository can currently support: post-training evaluation and rollout/trainer integrity, followed by an independent clean-clone reproduction or a narrowly scoped, externally useful systems contribution. A future model-learning claim still requires a new, genuinely viable task and a preregistered comparison of base, matched SFT and one RL method on independent task outcomes. This report does not establish an RL failure in general, coding-capability level, frontier-scale readiness, or external reproduction.
