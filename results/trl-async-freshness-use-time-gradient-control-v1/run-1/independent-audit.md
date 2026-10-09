# Independent audit — gradient control v1

Same-host independent review verified:

- protocol-pinned hashes for the reproducer, TRL base/candidate, candidate patch, Transformers Trainer, and Accelerate DataLoader;
- exact equality of baseline and candidate result JSON;
- all four arm-by-accumulation configurations reported CPU and exit code 0;
- GA=1 row 11 was trained at version 1 (lag 1), with raw returned-loss gradient 16 and masked-arm gradient 0;
- GA=1 final parameter was `0.09849999845027924` for queue admission and `0.0989999994635582` for masking, a difference of approximately `−0.000500001`;
- GA=2 row 11 was fresh (lag 0), with gradient 16 in both arms and equal final parameter `0.0989999994635582`.

Interpretation check: the counterfactual zeros only the positive-lag row's scalar loss contribution while retaining the batch and optimizer-step schedule. It does not establish a GRPO-compatible filtering/masking rule or a contract violation. The logged gradients are before Trainer's default clipping (`max_grad_norm=1`) and accumulation normalization; the parameter difference includes clipping and the learning-rate scheduler's schedule. This is a same-host agent review, not outside-human reproduction.
