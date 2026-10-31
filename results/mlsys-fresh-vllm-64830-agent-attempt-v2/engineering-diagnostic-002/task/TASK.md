# Task: accept structured tool arguments

At pinned upstream baseline `2af5f7ce8620115bf37fbbddf9817f083118df59`, the MLX-LM server's message normalization raises when an OpenAI-style function call supplies `arguments` as an already-decoded JSON object. Update the server so object-valued arguments are accepted while JSON-string arguments keep their existing parsing behavior.

Only edit `mlx_lm/server.py`. The evaluation checkout contains production Python files from the pinned baseline, but no Git history or upstream tests. Do not add dependencies. The change is evaluated against the selected behavior of MLX-LM PR #1904; the isolated grader exercises the message-normalization function contract, not the full HTTP/model-serving test fixture.
