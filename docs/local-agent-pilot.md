# Local CPU agent trajectory pilot

This pilot measured whether a small, locally cached instruction model could use a bounded file-editing tool loop to solve one pinned historical repair task. It was run offline on CPU without paid compute. The formal cohort did not produce a passing patch.

## Frozen setup

- Task: [HF rollout/learner probability parity](../benchmarks/historical/rvl_behavior_policy_parity/TASK.md), pre-fix source revision `e788f113ad6b246a361cd50a52eaf2867f2a66f8`, locked grader protocol v3.
- Model: [Qwen2.5-0.5B-Instruct snapshot `7ae5576`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), Apache-2.0. Exact local model files are hash-recorded in the [v2 protocol](../protocols/local_agent_trajectory_rvl_pilot_v2.json) and run outputs; weights are not vendored here.
- Runtime: macOS arm64, Python 3.12.12, PyTorch 2.9.1, Transformers 4.57.3, four CPU threads. All model parameters were placed on CPU; CUDA was unavailable and MPS was unused.
- Tools: list task files, read one file, search literal text, apply a small unified diff to an allowlisted source file, parse Python syntax, and finish. The agent received no shell, network, grader, or arbitrary code-execution tool.
- Budget: three fresh workspaces, seeds 54–56, at most 10 generations, 256 new tokens per generation, 2,560 tokens total and a 300-second soft wall limit checked between generations. Sampling used temperature 0.2, top-p 0.95 and top-k 20. One no-tool reminder was permitted.
- The model snapshot was already on the machine. No weights were downloaded during the runs; Hugging Face offline flags were enabled. No GPU, paid API or external compute was used. The host OS did not enforce network egress isolation.

The [v2 protocol](../protocols/local_agent_trajectory_rvl_pilot_v2.json) was frozen before the formal seeds. It records the model and source hashes, prompt, tool schemas, budgets and disclosed exploratory runs. The bounded runner is [`run_local_agent.py`](../scripts/run_local_agent.py).

## Formal result

| Seed | Outcome | Tool calls | Generated tokens | Wall seconds | Patch | Locked grader |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 54 | Read source; malformed diff rejected; reminder repeated the malformed diff | 4 | 330 | 57.241 | Empty | Rejected (36 failures) |
| 55 | Read source; placeholder patch rejected; reminder repeated it | 4 | 228 | 45.261 | Empty | Rejected (36 failures) |
| 56 | Read source, reread it, then stopped without a patch | 3 | 596 | 85.560 | Empty | Rejected (36 failures) |

All three runs made zero source edits and the locked grader rejected all three unchanged pre-fix workspaces. Seeds 54 and 55 attempted edits but omitted valid unified-diff file headers; seed 55's patch also contained a placeholder implementation. Seed 56 did not attempt an edit after rereading the source. The result is a negative tool-loop/usability observation for this model, prompt, tool contract and public historical task. It is not evidence about coding agents generally, successful task solving, novel discovery, or generalization.

## Invalidated and exploratory runs

The original v1 cohort (seeds 44–46) is retained in [`results/local-agent-rvl-pilot-v1/`](../results/local-agent-rvl-pilot-v1/) for audit, but excluded from model-outcome inference: its runner serialized assistant tool calls as ordinary text instead of the Transformers `assistant.tool_calls` structure. Its previous “no tool call” conclusion is invalidated.

Exploratory seeds 41–43 are private and excluded. Exploratory public seeds 51–53 are disclosed in the frozen v2 protocol and excluded from the formal cohort because the tool/path/prompt contract changed during development. They respectively exposed a missing path alias, repeated path guesses, and oversized replacement behavior while the edit tools were being redesigned. These exploratory observations motivated a separate formal protocol; they are not pooled with seeds 54–56.

The task is a public historical fix, so this pilot cannot establish novel discovery. A single task and model provide no evidence about generalization. The model process used offline loading and had no network tool, but the pilot did not enforce OS-level network isolation. The locked grader executes candidate source in a subprocess with the current user's permissions and is not a hostile-code sandbox.

## Raw evidence

- [Formal v2 aggregate](../results/local-agent-rvl-pilot-v2/summary.json)
- [Formal v2 SHA-256 manifest](../results/local-agent-rvl-pilot-v2/manifest.json)
- [Frozen v2 protocol](../protocols/local_agent_trajectory_rvl_pilot_v2.json)
- Invalidated [v1 summary](../results/local-agent-rvl-pilot-v1/summary.json) and [v1 manifest](../results/local-agent-rvl-pilot-v1/manifest.json)

Each formal `run-N.agent.json` preserves the model output, tool calls and results, initial prompt, model and source hashes, diff and runtime. Each `run-N.grade.json` preserves the locked grader output.

## Reproduce

Prepare each fresh candidate workspace before running the offline model:

```bash
python3 scripts/prepare_task.py \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --workspace /tmp/vare-agent-rvl-54
```

Install the optional packages in [`requirements-agent.txt`](../requirements-agent.txt) and make the exact model snapshot available locally. Then run with the snapshot path:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 \
python scripts/run_local_agent.py \
  --workspace /tmp/vare-agent-rvl-54 \
  --model-path /path/to/7ae557604adf67be50417f59c2c2f167def9a775 \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --instruction 'Read src/rvl_systems/hf_backend.py now. Implement the task in the prepared workspace. Use a minimal apply_patch, check syntax, and finish after a source change. Do not ask questions.' \
  --output /tmp/vare-agent-rvl-54.agent.json --seed 54 \
  --max-turns 10 --max-no-tool-retries 1 --max-new-tokens-per-turn 256 \
  --max-total-new-tokens 2560 --wall-seconds 300

python3 scripts/grade_task.py \
  --task-root benchmarks/historical/rvl_behavior_policy_parity \
  --workspace /tmp/vare-agent-rvl-54 \
  --result /tmp/vare-agent-rvl-54.grade.json
```

Repeat with fresh workspaces and seeds 55 and 56. Task preparation fetches the pinned source before the run; model and tokenizer loading require the local snapshot. Offline flags prevent those libraries from fetching files during agent execution.
