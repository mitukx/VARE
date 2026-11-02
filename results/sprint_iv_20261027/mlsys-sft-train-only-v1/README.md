# Train-only MLX-LM SFT feasibility result

This is a prospective, frozen 20-update LoRA run over the five VARE MLSys v1 training examples. The task Development campaign is separate. No held-out tasks, hidden tests, or Development trajectories are included.

The model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`, executed with MLX 0.32.4 / mlx-lm 0.32.2 on Apple Metal. The final adapter SHA-256 is recorded in `adapter-manifest.json`; adapter weights are kept outside this PR.

The training loss fell from 0.628 to 0.150 over 20 optimizer steps. This is only a train-objective diagnostic: there was no independent no-update comparison, hidden grading, held-out run, or capability-improvement claim. The stock MLX-LM command does not retain optimizer state.

`execution-protocol-v2.json` is the exact executed protocol (SHA-256 `2243ee5bf6273140172e015fdee9c5d2671749c61b4393bf49ea0f3fa14e9e9d`). `train.stdout.redacted.log` is a convenience rendering with terminal control codes removed and local paths normalized; the raw stdout/stderr remain in the local experiment tree. Verify every published file with `shasum -a 256 -c SHA256SUMS`.
