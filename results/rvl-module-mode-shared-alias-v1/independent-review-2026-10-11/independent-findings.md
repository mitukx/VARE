# Independent AI review: RVL shared-module rollback

## Scope and conclusion

I inspected only RVL baseline `27ebf7fe239d97504eeb960bf1407de220584dac` and candidate `7568c6e39015f18bec337174bacf39caac4f2477`, focusing on `src/rvl_systems/hf_trainer.py` and `tests/test_mini_lab_torch.py`. I independently derived and reproduced the shared-child rollback defect, and the one-line candidate fix addresses that failure for the captured per-module `training` flags. I found a documented semantic boundary around custom `Module.train()` overrides: direct flag assignment does not execute their arbitrary side effects. The existing snapshot stores only boolean mode flags, so this is outside the state it can restore; it remains a limitation to disclose.

## Failure mechanism derived from source

The baseline snapshots one `training` boolean per entry in `model.named_modules()`. PyTorch's module traversal deduplicates a shared module object, but `Module.train(mode)` recursively calls into descendants. During restoration, each saved entry is restored in traversal order. If a shared child is first visited through one parent and another aliased parent is restored later with a different mode, that later recursive call overwrites the child's saved flag. The resulting module graph no longer matches the snapshot; an active Dropout can therefore change the next forward.

The candidate keeps the snapshot and topology validation, then writes each unique module's `training` attribute directly. For ordinary PyTorch module flags this avoids the later parent's recursive propagation and restores the captured boolean state.

## Executed CPU checks

1. Ran the candidate's shared-child regression against candidate commit `7568c6e`: **pass**.
2. Ran that same candidate regression test against the pinned baseline source `27ebf7f` from an extracted baseline tree: **fail as expected**, with exactly `per-module modes` and `subsequent dropout forward` in the assertion's failure list. The test's other checks for model tensors, optimizer state, and CPU RNG did not fail on baseline; candidate passed all assertions, including those three state checks.
3. Ran a separate minimal PyTorch alias graph with both module registration orders. When the last-registered parent had a different mode from the shared Dropout, recursive `.train()` restoration failed exact flags and the next eval Dropout output in both orders; direct flag assignment passed both. A masking case where the final parent happened to match the shared child's mode passed under the baseline algorithm, confirming that test outcomes depend on the graph order and saved mode pattern.
4. Ran a synthetic `nn.Module` subclass whose `train()` override updates an extra attribute. Direct flag restoration restored the boolean `training` flags but left that custom attribute in its mutated state. This confirms the documented limitation. A search of RVL `src/` and `tests/` found no PyTorch `Module.train()` override; the sole `def train` hit is an unrelated agent training method.

The fresh baseline test required copying the candidate regression test into a temporary extracted baseline tree; no tracked source or test file was edited. The first attempt accidentally imported candidate source because the test module uses `src.rvl_systems` and the current checkout shadowed the extracted source. That invalid attempt is retained and marked in `baseline-import-misroute.note`; the corrected run is the one used above.

## Review judgement and boundaries

- The failure is real on the pinned baseline and the proposed direct-assignment change fixes the demonstrated failure.
- The regression exercises the actual tiny GPT-2 RVL trainer, a real AdamW mutation, injected post-update failure, and checks model/optimizer/CPU RNG/module flags/next forward.
- The separate ordering check shows both order directions and a masking case; this supports the mechanism rather than proving incidence in a default production model.
- The custom override result is a residual compatibility limitation. Direct assignment intentionally restores only stored flags, not arbitrary subclass behavior. No custom override was found in RVL source/tests; external Transformers model subclasses were not audited.
- CPU only. CUDA RNG, distributed execution, and real workload incidence were not tested. MPS was available but not used.
- This is an AI-agent review, not an independent human review. No GitHub action, comment, review request, or external notification was made.
