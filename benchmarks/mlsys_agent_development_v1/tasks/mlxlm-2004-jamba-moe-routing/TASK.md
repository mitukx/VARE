# Task: Jamba sparse-MoE routing probabilities

In `mlx_lm/models/jamba.py`, correct `JambaSparseMoeBlock` routing so selected expert outputs use probabilities normalized across all experts. Preserve top-k expert selection and make the probability dtype safe for mixed precision. Keep the change focused on the routing calculation.

Only edit `mlx_lm/models/jamba.py`. The candidate agent will not receive the trusted grader or the upstream repair.
