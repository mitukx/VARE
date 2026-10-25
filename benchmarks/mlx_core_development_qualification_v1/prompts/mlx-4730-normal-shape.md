# Task: requested output shape for `mlx.random.normal`

Make `mlx.random.normal` preserve the requested sample shape when `loc` or `scale` is broadcast. Reject parameter shapes that would enlarge the requested output, while continuing to accept broadcasts whose result is exactly the requested shape.

Keep the change focused on the implementation. Do not access the network or modify unrelated files.
