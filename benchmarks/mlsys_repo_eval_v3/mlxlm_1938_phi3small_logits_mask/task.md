# Phi-3-small dummy vocabulary logits

Inspect the model implementation and make its designated unused tokenizer
entries impossible to generate. The guarantee must hold for every batch item
and every sequence position, while ordinary model output keeps its normal
`batch × sequence × vocabulary` shape. Do not change model weights, tokenizer
IDs, or unrelated model behavior.

The grader uses a small randomly initialized model and checks both output shape
and the designated logits. No pretrained weights or network access are needed.
