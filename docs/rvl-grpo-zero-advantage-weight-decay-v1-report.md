# RVL GRPO zero-advantage AdamW behavior — v1

**Decision: retain the narrow reproduction as an observation of expected AdamW behavior, not as a demonstrated trainer defect or capability result.** A frozen CPU experiment reproduced the effect of AdamW's implicit default and showed that an explicit zero-decay default removes it on a fresh optimizer's first step. This v1 candidate changed ordinary training defaults; the compatibility-preserving v2 is the only current upstream candidate.

## Research question and novelty

When all responses in a complete same-prompt GRPO group receive the same reward, group-relative advantages and the policy-loss gradient are zero. Does RVL's trainer still change policy parameters?

This is a narrow optimizer-configuration observation in RVL revision `c7e646b043cb56e5ea3c2623bb8a61e065451f72`, not a new RL algorithm or a demonstrated correctness defect. The pinned `HFTTrainerConfig` had no `weight_decay` field and called `torch.optim.AdamW` without that argument. PyTorch AdamW therefore supplied its default `0.01`. With a zero gradient tensor and a fresh optimizer, AdamW still applies decoupled shrinkage:

```text
theta_after = (1 - learning_rate * weight_decay) * theta_before
```

At `lr=1e-3`, the one-step multiplicative factor is `0.99999`. This is expected AdamW behavior when weight decay is enabled. The wrapper did not expose the option, which is a configuration/API limitation rather than evidence that the optimizer update violated its contract. The v1 candidate exposed the parameter and used `0.0` by default, changing ordinary RVL training from its previous implicit `0.01`; that design was superseded by v2, which preserves the effective default.

The comparison is not evidence of a TRL bug. Current Transformers `TrainingArguments` declares `weight_decay=0.0`, while RVL directly instantiates PyTorch AdamW and inherited `0.01`; the two wrappers therefore have different effective defaults unless RVL passes the value explicitly ([Transformers training arguments](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py), [pinned RVL source](https://github.com/mitukx/Recursive-Verification-Lag/blob/c7e646b043cb56e5ea3c2623bb8a61e065451f72/src/rvl_systems/hf_trainer.py)).

## Frozen protocol

The protocol was frozen in VARE commit `4ef8f5ba4b115ea75c3e6ae59018cb493788ce37`, before base and candidate confirmatory runs. It pins RVL source revision and SHA-256, a tiny random one-layer GPT-2 model, one four-response group, CPU-only execution, PyTorch `2.9.1`, Python `3.12.12`, learning rate `0.001`, and seeds `17, 23, 29`. It includes:

- **Base:** constant reward `[1,1,1,1]`; require zero advantages, loss, gradients, and grad norm, effective decay `0.01`, nonzero parameter change, and exact agreement with the analytic decay formula.
- **Candidate default:** same constant-reward fixture; require zero effective decay and an identity parameter update.
- **Explicit-decay control:** candidate with `weight_decay=0.01`; require analytic-formula agreement.
- **Learning control:** rewards `[0,0,1,1]`; require finite nonzero gradients and a parameter update.

The frozen lock is [`rvl_grpo_zero_advantage_weight_decay_v1.lock.json`](../protocols/rvl_grpo_zero_advantage_weight_decay_v1.lock.json), canonical digest `9f5a0590209fc56e0d1892bbfa4e4fb2e012aeadbcfe3fddef5a11242a6e4448`. Its raw file SHA-256 is `f291298709cb64e423174947659e935186267a37f859f8203c6490c02d7f7b0d`.

## Results

| Arm | Seeds | Effective decay | Zero-reward gradients | Changed parameters | Max parameter delta | Formula error |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Pinned base, constant reward | 17, 23, 29 | 0.01 | 16 tensors present; 0 nonzero | 10 | `1.001358e-5` | `0` |
| Candidate default, constant reward | 17, 23, 29 | 0.0 | 16 tensors present; 0 nonzero | 0 | `0` | `0` |
| Candidate explicit decay, constant reward | 17, 23, 29 | 0.01 | 16 tensors present; 0 nonzero | 10 | `1.001358e-5` | `0` |
| Candidate default, mixed reward | 17, 23, 29 | 0.0 | 16/16 nonzero | 16 | `1.000047e-3` | Not applicable |

For the base run, the frozen runner reported loss and grad norm `0.0`, all four recomputed advantages `0.0`, and exact agreement with the analytic AdamW update for every seed. On the candidate default arm, all model state tensors were bitwise unchanged for all seeds. Explicit decay reproduced the base behavior exactly. In the mixed-reward control, a separate parameter-only recount confirmed 16 changed parameter tensors for each seed; the frozen runner's field named `changed_parameter_tensor_count` counts changed state-dict tensors and reports 17 because one changed entry is a non-parameter buffer. Treat the retained raw field as a state-tensor count; the separate parameter recount is reported here to avoid conflating it with parameters.

The next-token KL values on the base/explicit-decay runs are on the order of `1e-8`, including a small negative estimate from floating-point error. They are not interpreted as meaningful policy-distribution evidence. The mixed-reward KL is positive (`1.33e-4` to `2.73e-4`), but only serves as a fixture-level update check. All arms create a fresh optimizer and make one step; the experiment does not test zero gradients with nonempty AdamW moment state.

Raw outputs:

- [Base run](../results/rvl-grpo-zero-advantage-weight-decay-v1/base-run.json), SHA-256 `a519a7e95b13fce5ee2b390cbbb6b83b0fe32c7ccb5b3354210783244dcfd3c3`.
- [Candidate run](../results/rvl-grpo-zero-advantage-weight-decay-v1/candidate-run.json), SHA-256 `0f79c23e3994dfd609e41ca3f98538d2a593c473cbaab7fe6dca0c1a98d635f9`.
- [Unregistered development reproduction](../results/rvl-grpo-zero-advantage-weight-decay-v1-development.json) is retained separately from the frozen runs.

## Candidate change and checks

Candidate commit `e8a7276d1a5627cc2aef9e01526ca723c1eefd99` in local isolated RVL worktree branch `vare/grpo-zero-advantage-weight-decay` adds `HFTTrainerConfig.weight_decay=0.0`, validates finite nonnegative values, passes it to AdamW, and adds a regression test. The portable patch is [`rvl-grpo-zero-advantage-weight-decay-v1.patch`](../results/rvl-grpo-zero-advantage-weight-decay-v1/rvl-grpo-zero-advantage-weight-decay-v1.patch). It has not been merged or published upstream.

Checks completed on the isolated candidate worktree:

- Frozen base and candidate CPU harness: all predeclared criteria passed.
- Targeted regression via `python -m unittest tests.test_mini_lab_torch.TorchAcceptanceTests.test_zero_advantage_does_not_decay_policy_unless_configured -v`: passed.
- Full `tests.test_mini_lab_torch` module via `python -m unittest tests.test_mini_lab_torch -v`: 12 tests passed.
- Independent read-only audit: protocol, runner, source, candidate-test hashes and both raw runs matched; equations and outcomes independently checked. The auditor noted the count-field caveat above and the compatibility consequence of the new default.
- The adjacent original RVL checkout was left untouched; all candidate work is isolated in the clean worktree.

Reproduce from this VARE checkout with the pinned source and local candidate paths:

```bash
python scripts/run_rvl_grpo_zero_advantage_weight_decay_v1.py \
  --rvl-source ../../work/rvl-arc-update-smoke-pinned \
  --arm base \
  --output /tmp/vare-rvl-weight-decay-base.json

python scripts/run_rvl_grpo_zero_advantage_weight_decay_v1.py \
  --rvl-source ../../work/rvl-grpo-weight-decay-candidate \
  --arm candidate \
  --output /tmp/vare-rvl-weight-decay-candidate.json
```

The frozen runner refuses outputs that already exist. It verifies the protocol digest, runner hash, pinned trainer hash, candidate test hash, runtime version, CPU placement, and absence of CUDA.

## Claim boundary and decision

Established: on this pinned trainer and cold-start fixture, a zero policy-gradient step changes model parameters under AdamW's implicit `0.01` decay, matching the expected optimizer formula. Explicit zero decay gives an identity on the tested first step; the mixed-reward v1 control also updates under zero decay. V2 later preserved the prior default while exposing the choice.

Not established: that `0.0` is the uniquely correct default; that this changes real training quality, reward hacking, task success, or model capability; that the effect generalizes to resumed optimizer state or every optimizer/configuration; or that TRL has the same issue. The three seeds are deterministic fixture replications, not a statistical sample.

**Decision: STOP the v1 zero-default proposal; preserve it as design history and use the v2 compatibility candidate for upstream review.** Do not launch more model experiments for this question. The primary VARE gap remains independently measured, post-update task-success improvement.
