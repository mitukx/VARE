# Sequence-level DPO development v7 setup failure

The frozen v7 setup stopped 3.77 seconds after launch, before model inference or generation. It configured a generation batch larger than one, but the shared greedy decoder requires unpadded single prompts and rejected the padded batch. This is a harness/configuration failure, not an experiment result. The selected hash ranks 2720–2783 are retired and were not reused.

The failure bundle is [`results/cpu-lm-gsm8k-sequence-dpo-development-v7/run-1/`](../results/cpu-lm-gsm8k-sequence-dpo-development-v7/run-1/). The follow-up v8 lock uses fresh ranks and generation batch size one. Numeric rollout attachment also now uses the numeric verifier-label field when constructing rejected pairs.
