# TRL environment-column source replay

The public source intake in #208 includes [TRL #8520](https://github.com/huggingface/trl/pull/8520), a narrow correction to GRPO's signature-column contract. Losing the environment column may prevent a multi-environment trajectory from selecting the intended environment when unused-column filtering is enabled.

This prospective [source-only protocol](../protocols/trl_8520_environment_column_source_v1.json) pins the actual upstream merge parent, fixed revision, trainer blob, upstream regression-file blob, and Apache-2.0 license blob. The runner retrieves and validates these exact bytes, then extracts only the small signature initialization method. It refuses arbitrary expressions and does not import the upstream module. A separate in-memory oracle checks environment-only rows, mixed environment routing, retention of existing prompt/image columns and preservation of an already initialized signature. Three wrong controls exercise distinct failure modes.

The expected outcomes are baseline fail, known fix pass, all three mutants fail. This is a hypothesis until the pinned workflow completes. The oracle substitutes an in-memory column projection for Transformers Trainer; it does not execute the upstream full training test or certify an independent general-purpose grader. No task is allocated. Public pretraining exposure and complete historical family overlap are unknown. The frozen readiness decision and prior counts are unchanged. No model weights, inference, training, or held-out data are used.

## Measured outcome

The [pinned CPU source replay](https://github.com/mitukx/VARE/actions/runs/45604837382/job/139172936613) at `f8d6c81a3088940174c9fa1bbbea32cf67bba106` completed with PASS_SOURCE_REPLAY. The downloaded trainer, upstream regression file and Apache-2.0 license matched the frozen Git blob IDs.

| Arm | Passing cases | Disposition |
| --- | ---: | --- |
| Baseline | 2/4 | FAIL |
| Known fix | 4/4 | PASS |
| Wrong plural | 2/4 | FAIL |
| Drop image | 3/4 | FAIL |
| Overwrite existing | 3/4 | FAIL |

[Original ZIP](../results/trl-8520-environment-column-source-v1/source-replay.zip) contains the byte-exact result. Archive SHA-256 `ff10fabdb1b4b1fd1224b06299fd56ade4a8e798176bbf5814f5eabca0a505f3`; contained result SHA-256 `25b73032f007608f4d1edb0ffe242758de83c00ca467c6701f76fab5c3ef11d0`. The offline evidence regression verifies these digests and the outcome inventory. This is an isolated method/source replay, not the complete upstream training regression or certification for a blind task. No new eligible family is credited. Automatic repeat is disabled.
