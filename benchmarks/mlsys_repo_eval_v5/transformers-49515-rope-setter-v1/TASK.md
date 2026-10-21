# Legacy RoPE configuration compatibility

Legacy RoPE configuration dictionaries remain accepted through the compatibility setter and should be normalized before rotary embeddings are initialized. A small Llama configuration assigned a legacy scaling dictionary currently fails during initialization. Preserve the legacy input form while making initialization succeed with the intended scaling type.

The local source checkout and public smoke test are available in the workspace. Do not use network access, pretrained checkpoints, or Git history.
