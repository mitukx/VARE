# Flagship A100 run

`notebooks/flagship_a100.ipynb` and `scripts/run_flagship.py` run two CPU-tested loops on one CUDA GPU. No model result is retained in this repository. A finished GPU JSONL on Drive is the measurement of that run.

## Part A

Five arms train, or in the base arm only generate, on the integer seed-split train rows: correctness, Bernoulli random, incorrect, format, and base. The random arm uses the difficulty-probe accuracy as its rate and does not read the completion. The incorrect arm pays 1 when the strict `#### <integer>` line is wrong. Format pays 1 when that line parses. The base arm logs the same train metrics and does not step the optimizer. Reference KL is logged with coefficient 0. The base arm records KL 0 and does not run the reference model.

The models are Qwen2.5-1.5B-Instruct (`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) and Llama-3.2-1B-Instruct (`9213176726f574b556790deb65791e0c5aa438b6`). If Llama weights cannot be loaded, the run prints the error and uses SmolLM2-1.7B-Instruct (`31b70e2e869a7173562077fd711b654946d38674`). The update is last-layer LoRA, rank 8, on `q_proj` and `v_proj`, in bf16. Generation is Hugging Face `generate` with left padding. One torch seed covers a batch.

A 16-row greedy probe on dev difficulties 2, 3, and 4 picks the difficulty. A rate in `[0.2, 0.5]` wins. Otherwise the rate closest to 0.35 is used and the log says `calibration_inside_band: false`. The confirm split is not loaded. Held-out items are a seeded GSM8K test subset. The primary score is the strict final line. A last-integer match is recorded and does not decide.

FULL starts at 100 steps, group 8, two seeds, and an eval every 20 steps on dev plus 200 held-out items. With the unmeasured prior that grid does not fit, so the planner lowers it to 60 steps and leaves the held-out set at 200. QUICK is one model, one seed, 30 steps, group 8, and an eval every 6 steps on 16 dev and 64 held-out items, which is about 17 minutes on the same prior. QUICK writes `throughput.json`. FULL reads it when the device field is `cuda` and reduces steps by 20, down to 40, until the projected time is within 2.5 hours. Part B and loads are reserved 25 minutes in that projection. If the floor is still over, held-out size drops by 40 down to 40 and the log says so. `run.json` then freezes the grid. The unmeasured prior is 0.35 s/train sample and 0.20 s/eval sample. That prior is not a timing measurement.

## Part B

Qwen2.5-1.5B-Instruct proposes recomputable integer problems and trains for a few rounds. One source pays the recomputed answer. The other pays the model's own verifier: a final `#### <integer>` line accepts the candidate when the integers match. The verdict is not executed. The dev split has 32 tasks, so the loop passes `min_eval_examples=32` and turns the paired bootstrap off. Confirm tasks are not added.

`tests/test_flagship.py` feeds the gate a case where train reward rises, dev accuracy falls, and the false-accept rate is 0.5. The gate rejects it. A GPU round records the gate decision for that short run, including a rejection when the measured candidate does not clear the same rule. The figure is self-reward against dev accuracy with accept or reject marks.

## Resume

Train metrics, eval rows, and round decisions are appended to `metrics.jsonl` and flushed. Optimizer steps write a checkpoint. The base arm writes a marker. A gap in the step list, or a metric row whose checkpoint is missing, stops the run. Re-running the same output directory continues. Changing mode or plan version in that directory stops the run.
