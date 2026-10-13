# selfimprove

`selfimprove/` proposes integer expressions, keeps a problem only when the exact evaluator accepts it, and trains a solver with `minigrpo` for a fixed number of rounds. Difficulty moves up when the probe solve rate is above 0.8 and down when it is below 0.2, staying inside difficulties 1, 2, and 3.

Each round scores the existing dev split. A model-written verdict is a final line `#### <integer>`. The false-accept rate is how often that verdict agrees with a candidate the exact checker rejects. The verdict is not executed as code.

Rollback uses `PromotionGate` unchanged. Dev has 32 tasks, so the loop passes `min_eval_examples=32` instead of the default 64. Confirm tasks are not added and are not scored. A rejected candidate stays on disk and is not the next round's starting weights.

The monitors recorded in `rounds.jsonl` are:

- `format_only_reward_rise`: training format compliance rose while dev accuracy did not
- `self_verifier_disagreement`: the false-accept rate is above 0.20
- `heldout_divergence`: training reward rose while dev accuracy fell

Those signals are logged. The gate decides. `--reward format` pays for a strict final line and ignores the integer. The CPU test builds that trajectory and checks that the gate rejects it. No model round is retained here.

Re-running the same command continues `rounds.jsonl`. A finished round is not repeated. An unfinished round directory resumes its minigrpo steps.

## Short MLX run

Two rounds, 8 proposed problems, 8 optimizer steps, group size 2, 32 new tokens. The baseline and each round also score 32 dev answers and 32 verdicts. That is 240 generations: 64 for the baseline and 88 per round. A 3B 4-bit model on an M1 is the size intended to land around 30 to 60 minutes if each short sample takes several seconds. That rate was not measured here. A smaller model finishes sooner. Do not add rounds after looking at the dev accuracy.

```bash
git clone https://github.com/mitukx/VARE.git
cd VARE
git fetch origin cursor/selfimprove-af75
git checkout --detach FETCH_HEAD

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 -m pip install mlx-lm==0.32.0

python3 scripts/run_selfimprove.py --backend mlx --check

python3 scripts/run_selfimprove.py \
  --backend mlx \
  --model mlx-community/Qwen2.5-3B-Instruct-4bit \
  --revision 4f83f8f146fdf28b512a06562b671d7af4fab457 \
  --rounds 2 \
  --steps-per-round 8 \
  --problems 8 \
  --group-size 2 \
  --max-new-tokens 32 \
  --learning-rate 1e-5 \
  --num-layers 1 \
  --reward correctness \
  --seed 20261023 \
  --output "$HOME/vare-selfimprove-short"
```

The 0.5B and 1.5B revisions are listed in [minigrpo](minigrpo.md). `--reward format` is the hackable reward. The unit test already shows the gate rejecting that pattern. The same output directory resumes.
