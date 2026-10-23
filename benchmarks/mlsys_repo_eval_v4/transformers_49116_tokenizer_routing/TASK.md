# Fix Qwen tokenizer routing for Hub configuration variants

Several Qwen model configurations identify the newer Qwen3.5 tokenizer format while their Hub metadata names an older tokenizer class. Make automatic tokenizer selection consistently choose the tokenizer implementation registered for the model configuration, including the affected mixture-of-experts and text-only variants.

Keep the change in `src/transformers/models/auto/tokenization_auto.py`. Preserve existing routing for unrelated model types. The workspace contains the pinned Transformers source and a small public regression test. Use only the allowed source file and the provided public test; do not use network access, model checkpoints, or Git history.
