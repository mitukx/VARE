# Operator-authenticated RVL reward admission (E0 control-plane correctness)

## Why this change exists

The RVL replay reader previously used `bool(metadata["trusted"])` and
therefore silently treated `"false"`, `"0"`, and other nonempty strings as
trusted evidence. Separately, `RVLOuterPlanner` allowed *untrusted* ready
verifier failures to increase curriculum priority. Thus a malformed or
unverified reward could affect which post-training task family received more
budget even when the policy/verifier freshness windows passed.

This change implements a narrow **testable reward-provenance boundary** for
the outer planner. It is not a new GRPO optimizer, autonomous RSI result,
hardware benchmark, or demonstration of real-model quality gains.

## New contract

`RVLTokenReplayReader.ready_experiences` now accepts only a literal Python
`bool` for the metadata `trusted` field and its fallback argument. No
truthy-string coercion is allowed. Note that even a literal `True` remains a
**self-reported SQLite field**, not cryptographic evidence.

`RewardEvidenceRoot` represents a secret-key, operator-pinned signing
context. The trusted independent scorer signs a canonical envelope for a
single RVL group member, including:

- unique operator-issued study ID; frozen protocol and evaluator SHA-256;
  frozen verifier configuration SHA-256;
- original rollout group ID/member index/size; unkeyed source behavior hash
  and exact current generation-content digest;
- task ID, prompt, response, policy ID/version, verifier ID/version, actual
  ready score, pass/fail, and literal trusted flag.

The original RVL behavior hash is only a consistency check and cannot
authenticate a reward. The signer must independently score the actual
generation and must not be callable by candidate-side code. Receipts are
stored in an **operator-owned sidecar**, keyed by a digest of group ID and
member index. They are not copied from writable replay metadata.

The planner now always excludes declared-untrusted fresh records from
reward-based curriculum amplification. In opt-in authenticated mode,
`reward_manifest=RewardEvidenceManifest(root, receipts)` additionally
requires a matching HMAC receipt for each eligible group member. With
`require_authenticated_rewards=True`, absence of a manifest is an error,
and **any** fresh ready row lacking eligibility produces a training freeze
decision, because VARE does not yet enforce the signed-only subset inside
RVL's downstream trainer. Complete signed membership is required for a
nonfrozen strict plan.

## Minimal trusted control-plane usage

```python
from vare.config import LagConfig
from vare.controls import RVLOuterPlanner
from vare.reward_evidence import RewardEvidenceRoot, RewardEvidenceManifest

# The evaluator must have independently adjudicated each score before
# invoking root.attest. Never share this HMAC key with the model or agent.
root = RewardEvidenceRoot(
    key=operator_key,               # 32+ random bytes, stored securely
    run_id="experiment-unique-2026",
    protocol_sha256=pinned_protocol,
    evaluator_sha256=pinned_grader,
    verifier_config_sha256=pinned_verifier_config,
)
# This mapping is built and held OUTSIDE the writable RVL replay database.
# The grader, not the learner, issues the signature for each scored member:
receipts = {root.record_key(e): root.attest(e) for e in independently_scored}
manifest = RewardEvidenceManifest(root, receipts)

plan = RVLOuterPlanner(LagConfig()).plan(
    rvl_reader,
    current_policy_version=active_policy_version,
    current_verifier_version=active_verifier_version,
    reward_manifest=manifest,
    require_authenticated_rewards=True,
)
if plan.control.freeze_training:
    # The outer orchestrator must enforce the control decision before
    # invoking any learner/optimizer, including RVL's own training path.
    ...
```

In actual deployment, the signer and any key-backed verifier must belong to
independently protected processes or services. HMAC verification gives the
controller signing ability as well; production multi-host architecture should
consider public-key signing and separate issuance/audit registries. Operator
must pin keys/identities before running or reading task outcomes; do not mint
receipts for historical caller-provided rewards without independent regrading.

## Last-mile learner admission in `CapabilityLoop`

`CapabilityLoop(training_evidence_authorizer=..., require_authenticated_training=True)`
now rejects a selected replay batch before calling `train_candidate` unless
**every** member passes a trusted use-time authorization. The callback is
evaluated after current policy/verifier freshness recomputation and after
whole-group replay selection; a failed or malformed return, or an exception,
aborts the entire update (no partial GRPO group training). Operator code can
supply `manifest.verified` when its `Experience` identities match the
independently attested RVL replay rows:

```python
# When constructing the outer-loop CapabilityLoop, not inside candidate code:
loop = CapabilityLoop(
    hooks=hooks, verifier=verifier,
    training_evidence_authorizer=manifest.verified,
    require_authenticated_training=True,
    # other configured controls...
)
```

This is an *explicit, opt-in gate*. It does **not** automatically wrap
RVL's standalone trainer. If RVL training is launched outside
`CapabilityLoop`, its operator must enforce the same attestation contract
at the actual training-consumption boundary. A trusted HMAC validator is
necessary but not sufficient: a compromised grader can still sign bad reward.

## Negative controls and reproducibility

Run:

```bash
python3 -m pytest -q tests/test_rvl_reward_attestation.py \
    tests/test_controls.py tests/test_rvl_integration.py \
    tests/test_training_evidence_admission.py
```

The adversarial fixtures attempt truthy-string trust escalation, training
curriculum inflation from untrusted rewards, deletion of all receipts, partial
attestation coverage, changing a reward in SQLite after signing, replaying
receipts under different task/group/evaluator/run/key, and stale-but-signed
verifier evidence. All tests are CPU-only, use fake replay data, and require
no GPU, network access, or model downloads.

Evidence level **E0**: code path and generated test outcomes only. A
successful regression means the tested replay and planner can reject the
specified corrupt evidence; it does not show lower actual reward hacking,
better model performance, or signer/host compromise resistance.

## Important limits and future work

- The trust bit is still declarative if the planner is called **without**
  `reward_manifest`; only strict mode yields cryptographically checked
  reward evidence, and it is opt-in for backwards compatibility.
- This API does not alter RVL's replay writer or trainer. `freeze_training`
  is a decision to be enforced by the trusted orchestrator; it does not
  itself stop arbitrary standalone RVL learner processes. The next
  integration should enforce signed membership immediately before actual
  learner consumption, including multi-member GRPO group atomicity.
- The sidecar receipt is scoped to group/member and run, not single-use
  optimizer consumption. An independent monotonic replay/consumption
  register is needed to prevent reusing an otherwise valid signed group
  across multiple optimization windows.
- A verifier signed by an isolated service can still be *incorrect*. This
  authenticates provenance, not reward accuracy. Independent adjudication,
  calibrated disagreement and matched-budget held-out experiments are
  required to measure whether the intervention improves RSI control.
- Do not run against retired confirmation cohorts or re-score existing
  negative studies in place. Freeze any future E3/E4 protocol first.
