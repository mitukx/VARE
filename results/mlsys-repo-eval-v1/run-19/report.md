# Run 19 — Qwen3-4B MLX-LM #1904 Train attempt

**Decision: PASSED, narrowly.** One local base-model agent attempt edited the pinned MLX-LM source, and the independent v12 grader passed both the structured-object case and the JSON-string control.

## Frozen setup

- Protocol: `mlsys-task-model-attempt-mlxlm-1904-v1`, frozen before inference; SHA-256 `7bdef7ee8210dfbd25b3ca811d8bb7c1450b97e4c6015e39d653879aee51ad3d`.
- Task: `mlxlm-1904-tool-arguments`, upstream `ml-explore/mlx-lm#1904`, assigned **Train** in the frozen qualification catalog. Only `mlx_lm/server.py` was editable.
- Source baseline: `2af5f7ce8620115bf37fbbddf9817f083118df59`; baseline file SHA-256 `57392576de114af80ede723976b9dd156dc234f856b9943b8b44b8b78bc27f79`.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` at revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`; weights SHA-256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`.
- Runtime: Python 3.12.12, MLX package metadata 0.32.4, MLX-LM 0.32.2, Apple Metal. The controller itself reports `mlx` version as `unknown`; the pre-run package metadata recorded 0.32.4.
- Greedy decoding, seed 1904; limit 12 turns, 256 tokens per turn, 2048 total tokens, 4096 context tokens, 300 seconds.
- Paid inference/API spend: $0. No model training or policy update.

## Observed result

- Agent stop: `agent_finished` after 5 tool turns and 302 generated tokens in 31.679 seconds.
- Edit applied: `True`. Diff SHA-256 `e95091b6c6ff482234360c01e59ef8a31cd5fd1ff2956ea8d47872decc5d8419`; source SHA-256 after the edit `58facacda929918a0ccb0cd14e22b38d0b80162dea54106b6e023e08679bcbfc`.
- Independent grader call count: 1; feedback returned to the agent: `False`.
- Cases: `dict_arguments` passed and `json_string_control` passed. Overall grade: **PASS, 2/2**.
- Peak MLX memory reported: 3429948504 bytes; process max RSS: 2617196544 bytes.
- A launch issue was encountered before inference: a linked Git worktree exposed `.git` as a pointer file, which the pinned workspace reader refuses. The model snapshot loaded, but no prompt was rendered and zero tokens were generated. The event is preserved in `launch-error-pre-inference.*`; the authorized single inference then ran from a verified source-only workspace with a real `.git` directory. This did not consume an inference attempt.

## Interpretation and limits

This is one independently graded trajectory on an already assigned **Train** task. It demonstrates a narrow task repair by the pinned base policy and current controller. It is not a fresh Development result, a generalization estimate, SFT or RL evidence, a policy update, or an algorithmic advantage. The task is a public upstream PR, so the model's pretraining exposure is unknown. The isolated grader covers only the message-normalization function contract and does not run the original HTTP/model-serving fixture or upstream test suite. The agent's source hash is not the catalogued known-fixed source hash; the claim rests on the frozen independent grader's two passing cases, not patch identity.

The one model exposure is appended in `exposure-append.json`. The earlier queue audit and Stage 2/3 ledger remain immutable; no Held-out task or test was opened.
