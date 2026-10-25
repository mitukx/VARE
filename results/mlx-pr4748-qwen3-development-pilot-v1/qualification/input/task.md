# Development task: preserve triangular inversion semantics under batching

When a function that performs triangular matrix inversion is mapped over a batch, some inputs produce different results from applying the same operation to each matrix directly. Correct the MLX vmap behavior so the mapped operation preserves the requested upper- or lower-triangular mode. Keep the change focused on `mlx/primitives.cpp` and do not use the network.
