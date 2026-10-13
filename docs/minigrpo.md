# minigrpo

`minigrpo/` is a small GRPO trainer for the integer seed-split train rows. After the last optimizer step it scores the dev split with the strict `#### <integer>` line. The confirm split is not loaded and is not scored.

The loss uses an intact group, a group-relative advantage, a PPO clip, and an optional reference KL (Schulman k3). Dr.GRPO mode (`--dr-grpo`) uses `reward - mean` and sums token losses instead of dividing by response length. Each training step appends reward, entropy, response length, KL, and format compliance to `metrics.jsonl`. Re-running the same command on the same output directory continues from the last contiguous checkpoint.

No model run is retained in this repository. A CPU smoke checks the loss and the metric log. That smoke is not a held-out gain.

MLX and PyTorch are optional. `--check` does not import them. The PyTorch path is there for a single CUDA GPU. It uses the same LoRA rank, scale, and last-layer `q_proj`/`v_proj` targets. It is not a numerical replicate of the MLX run.

The only measured Apple Silicon anchor in this repository is not a GRPO step. Plan a first run as a budget, and stop from the JSONL if it runs long. The same command resumes.

## Models

| Backend | Repository | Revision observed 2026-10-13 |
| --- | --- | --- |
| MLX 0.5B 4-bit | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3` |
| MLX 1.5B 4-bit | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `8b403126fc14f14cfc99bb4cfa72ecbc129ea677` |
| MLX 3B 4-bit | `mlx-community/Qwen2.5-3B-Instruct-4bit` | `4f83f8f146fdf28b512a06562b671d7af4fab457` |

The revisions are the hub commits read on 2026-10-13. Pass `--revision` so the run records that pin in `run.json`.

## Short MLX run

This command is the short first run: 40 steps, group size 4, 32 new tokens, one LoRA layer, no reference model. Forty groups plus one dev pass of 32 prompts is 192 generations. A 3B 4-bit model on an M1 is the size intended to land around 30 to 60 minutes if each generation-plus-update takes on the order of 10 to 20 seconds. That rate was not measured here. A smaller model finishes sooner. Do not add steps after looking at the dev accuracy.

```bash
git clone https://github.com/mitukx/VARE.git
cd VARE
git fetch origin cursor/minigrpo-af75
git checkout --detach FETCH_HEAD

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m pip install mlx-lm==0.32.0

python3 scripts/run_minigrpo.py --backend mlx --check

python3 scripts/run_minigrpo.py \
  --backend mlx \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --revision 4f83f8f146fdf28b512a06562b671d7af4fab457 \
  --steps 40 \
  --group-size 4 \
  --max-new-tokens 32 \
  --epochs 1 \
  --learning-rate 1e-5 \
  --num-layers 1 \
  --kl-coef 0 \
  --seed 20261022 \
  --output "$HOME/vare-minigrpo-short"

python3 -m minigrpo.plot \
  --metrics "$HOME/vare-minigrpo-short/metrics.jsonl" \
  --output "$HOME/vare-minigrpo-short/metrics.svg"
```

`--epochs 2` runs a second pass over the same group, which is when the clip can bind. The default is one pass. `--kl-coef 0.02` loads a second frozen copy for the reference KL. `--dr-grpo` switches the normalization. Swap the model and revision for the 0.5B or 1.5B rows above to use those checkpoints. The same output directory resumes.
