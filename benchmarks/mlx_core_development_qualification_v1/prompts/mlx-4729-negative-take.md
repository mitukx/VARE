# Task: negative integer indexing in `mlx.core.take`

Correct the scalar integer-index overload of `mlx.core.take` so negative indices select the corresponding element from the end of the selected dimension. The flattening overload must follow the same behavior, and existing positive-index behavior must remain intact.

Keep the change focused on the implementation. Do not access the network or modify unrelated files.
