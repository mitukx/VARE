# Integer seed split v1

Protocol id: `integer-seed-split-v1`.

`benchmarks/heldout/integer_seed_split_v1/splits.json` is the committed train, dev, and confirm seed list. Its SHA-256 is `4544a9a852f615f7cf9c52d7a4826ed70ea6c5f57829754ac2e10f3c1fea4287`, stored in `protocol.json`. The same master seed `20261014` regenerates that list. Confirm seeds are the `confirm_seeds` array in the file, not a range that can be extended after a result.

Difficulty is frozen at 2: `a + b - c`, with each split drawing integers from its own band. Whitespace token 3-grams of the expressions do not cross splits. None of the expressions are in the exact-integer v1 pack. No model output is attached. The file does not authorize an update.
