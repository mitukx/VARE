# ARC-Challenge CPU GRPO update-cost smoke v2

**Gate: PASS for designing a separate learning comparison.** This is one task-specific optimizer/checkpoint execution, not evidence that training improves ARC task success.

## Question and frozen setup

Can the pinned VARE → RVL GRPO path perform one real CPU update on ARC-Challenge training data, restore its incumbent exactly, and reload the candidate to generate answers on untouched validation items?

Protocol v2 was frozen in commit `0c3055e` before model inference. It pins the Qwen2.5-0.5B-Instruct revision, the RVL source revision, ARC dataset revision and parquet hashes, 8 fresh hash-ranked train candidates, 8 validation items disjoint from all base screens and the failed v1 smoke, seeds, reward parser, optimizer settings, resource caps, and source hashes. A maximum of one optimizer step was allowed. Validation exact-match was descriptive only and did not control the gate.

The v1 attempt is retained at [`run-1`](../results/qwen-arc-grpo-update-smoke-v1/run-1/). It generated base answers for its 8 validation items, then failed before training because the runner read `group_size` from the wrong protocol section. Those 8 items are excluded from v2. Two earlier v1 preflight failures are preserved separately; neither reached inference. V2 changed no prompt, parser, reward, or acceptance threshold based on the v1 outputs.

## Result

- The first frozen ARC train candidate produced rewards `[0, 0, 1, 0]`; VARE passed its four-member group to the pinned RVL GRPO trainer.
- Exactly **one** optimizer step ran. All **290/290** gradient tensors were finite and nonzero; **291** parameter tensors changed, with maximum absolute delta `1.91e-6`.
- VARE restored the incumbent exactly after the candidate update. The candidate checkpoint and tokenizer reloaded with matching parameter fingerprint and prompt token IDs.
- On the 8 reserved smoke items, base exact-match was **4/8** and post-reload exact-match was **3/8**; parse rate was **87.5%** and **100%**, respectively. These small-sample before/after values are descriptive only; they do not establish improvement or degradation.
- CPU wall time was **38.23 s**, peak RSS **10,747 MiB**, with no GPU or paid service.
- Frozen gate: **pass**. Same-host independent implementation audit: **58/58 checks passed**.

The raw bundle is [`run-1`](../results/qwen-arc-grpo-update-smoke-v2/run-1/), including sampled token IDs and ephemeral checkpoint-file hashes. The checkpoint itself was temporary and is not retained. Transformers emitted a tokenizer-regex warning while loading the saved tokenizer; exact token IDs matched on all frozen prompts, which does not validate arbitrary text. The audit output's free-text claim limit mistakenly says “16-item”; the protocol and its `n_validation=8` field, per-item checks, and this report show that eight items were used. No metric or gate depends on that stale sentence.

## Interpretation and next decision

This passes the ARC-specific update/save/reload feasibility gate. It establishes neither policy improvement nor a new algorithmic result. The observed one-item exact-match decrease is not an efficacy estimate. The model and task remain eligible for one separately frozen comparison of base, matched supervised learning, and GRPO across multiple seeds on untouched task-success examples. The failed v1 items and the v2 smoke items must remain excluded. The ARC test split has not been opened.

## Reproduction

The exact author-host commands are retained in [`commands.txt`](../results/qwen-arc-grpo-update-smoke-v2/run-1/commands.txt); replace local cache paths with the corresponding pinned artifacts on another host. Use the pinned RVL commit and model/dataset revisions from the protocol. The independent auditor reconstructs the selected rows, prompts, decoded responses, exact rewards, group-relative advantages, metric summaries, and frozen gate without rerunning optimization.
