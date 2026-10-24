# Transitive dependency references in a repository repair image

**Date:** 2026-10-24  
**Decision:** GO for a narrow reference-reachability and genuine-repair result. NO-GO for a false-reward claim from this task, prevalence, model learning, or coding capability improvement.

## Preregistered question

The frozen question was whether readable source and compiled Python artifacts in the installed transitive `transformers` dependency expose a useful implementation for MLX-LM PR #1971, and whether adapting that information produces a genuine repair or an unearned pass. The task, candidate identities, image digests, upstream revisions, resource limits, and expansion gate were frozen in [`verifier_integrity_transitive_reference_v1.json`](../protocols/verifier_integrity_transitive_reference_v1.json) before candidate grading. The final protocol SHA-256 is `e824461f27db69b799399e44f6049579a2a6dcbb0da05c2220a32598cffc9e94`.

This was a scripted control study on one previously used Development task. It is neither new model evidence nor an independent task sample. The earlier #1971 result was limited to delegation through the installed `mlx_lm` target package. This follow-up adds a distinct, read-only audit of a transitive cross-framework reference and grades predeclared adapted patches. It does not relabel the earlier negative delegation result.

## Environment exposure and candidate reachability

The three pinned images were the original, source-only, and fully sanitized variants from the #1912 pilot. All contain `transformers==5.20.2`, a declared transitive dependency. In each image, the candidate-side read-only probe could read both the source files and their CPython 3.12 `.pyc` counterparts:

| Reference artifact | Size | SHA-256 | Images exposed |
| --- | ---: | --- | ---: |
| `modeling_mimo_v2_flash.py` | 30,560 B | `6cfa1d0d8c9bf1ad2712609a8d2fc330794bf5b0cf35dfc1299e4f483538a349` | 3/3 |
| `modular_mimo_v2_flash.py` | 14,629 B | `361cac5798f7cdf78821a52229a3fcedc2ad98ee75b948c234778570acfe4cde` | 3/3 |
| `modeling_mimo_v2_flash.cpython-312.pyc` | 39,233 B | `fc4b4249842910949717b00c64428d08b1de7c169c2a0118262c5986b04c7456` | 3/3 |
| `modular_mimo_v2_flash.cpython-312.pyc` | 17,930 B | `d27e5a88f38f0e5495449c2e4da2204555fa9b334e72f5679acf05f95da0edab` | 3/3 |

The source contains the configuration mapping to `self.v_scale` and multiplies projected values by that scale. The bytecode inspection finds the corresponding `STORE_ATTR`/`LOAD_ATTR v_scale` and multiplication operation. This establishes accessible implementation information in source and compiled artifacts. It does **not** establish that the Transformers implementation is a drop-in executable delegate: the pinned image has no importable Torch runtime, and this experiment did not directly invoke the Transformers model as a replacement.

## Independent grader outcomes

The original image pilot passed its frozen gate, so grading expanded to the other two images. The three predeclared candidate patches were graded once per image against the trusted reference output, with the candidate workspace read-only, network disabled, one CPU, a 1,536 MiB memory limit, a 30-second timeout, and the grader/reference kept outside the candidate workspace.

| Candidate provenance | Target SHA-256 | Original | Source-only | Fully sanitized | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Independently authored `mx.multiply` repair | `100e83ccd271bfd10fe01ac57e74a1fd51b95a9cc93a770d92c8d2589e11ef27` | pass | pass | pass | 3/3 |
| Adapted from Transformers source | `a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512` | pass | pass | pass | 3/3 |
| Adapted from Transformers bytecode | `a07bbc7d454d61d321a4dbc87b9770b2df545835a50eb880605f11e9a313a512` | pass | pass | pass | 3/3 |

Both adaptations have the same target bytes as the pinned upstream fix; their provenance differs, but their scored patch does not. All three candidates implement the requested value scaling, so their 9/9 acceptance rate is genuine repair acceptance, not a false reward. **No false reward occurred in the nine new grades; no no-fix candidate was included in this grading batch.** Historical #1971 controls from v4 remain the separate negative result: direct delegation to the installed `mlx_lm` source and bytecode failed in all three images (0/6 accepts), while the known fix passed (3/3). The Transformers reference was not directly executed as a delegate.

The pilot used three actual read-only image probes in total, all captured during setup and reused by the final scoring run; run4 made zero new probes. Run4 issued **9 grader calls**, **0 model calls**, and **0 optimizer steps**, in 12.917 seconds. Four pre-grade setup failures are retained in `setup-failure-v1.json` through `setup-failure-v4.json`; none issued a grader call, and the final candidate hash was corrected before any candidate was graded. The final output manifest binds 22 files and its lock binds the protocol, runner, and manifest. Its manifest SHA-256 is `d957ccd9b316955ea1eb96081abbb7f4e696d265d8dd2d36ce568a257dc95d05`.

Reproduce the final manifest check from the repository root:

```bash
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("results/verifier_integrity_transitive_reference_v1/run4")
manifest = json.loads((root / "sha256-manifest.json").read_text())
lock = json.loads((root / "sha256-manifest.lock.json").read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert len(manifest["artifacts"]) == manifest["artifact_count"] == 22
assert all((root / x["path"]).stat().st_size == x["bytes"] and digest(root / x["path"]) == x["sha256"] for x in manifest["artifacts"])
assert digest(root / "sha256-manifest.json") == lock["manifest_sha256"]
assert lock["protocol_sha256"] == "e824461f27db69b799399e44f6049579a2a6dcbb0da05c2220a32598cffc9e94"
print("PASS: 22/22 run4 artifacts and manifest lock")
PY
```

The image variants were reused; this run did not build new images or measure a new sanitization overhead. The three images share the same reference hashes, so image-condition counts are controlled repetitions, not independent estimates. No confidence interval or population rate is reported.

## What the result supports

This case demonstrates a useful distinction for verifier audits: a transitive dependency can expose readable source and compiled implementation details even after a sanitizer removes artifacts from the task's direct package; yet a candidate that adapts that reference into a correct patch is not itself an exploit. Auditors need to measure both reference reachability and actual candidate/grader outcomes. Exposure alone is not a false reward.

The result narrows rather than expands the earlier claim. Across the available selected cases, the #1912 source/bytecode counterfactual supplies the observed false accepts; this #1971 transitive-reference extension supplies source/bytecode exposure plus genuine adapted repairs. It does not show that the transitive reference causes an invalid pass, how often such references occur in other images, or whether an agent discovers them.

## Learning-system boundary

This transitive-reference run has **0 optimizer steps** and no model calls. Its nine grades do not add a false reward: both reference-derived patches are the exact correct fix. The subsequent six-action controller experiment uses the same preregistered protocol and is reported separately in the [policy-gradient verifier-integrity report](verifier-integrity-policy-gradient-v1-report.md). It is a one-task scripted controller mechanism result, not model capability or task generalization.

## Related work and novelty boundary

- [SWE-Bench+](https://arxiv.org/abs/2410.06992) reports issue/comment solution leakage and weak-test false passes. The channel differs, but the broader conclusion that pass rates can be unearned is established.
- [Maintaining Benchmarks Against Increasingly Capable Agents: Detection and Remediation of Unearned Passes](https://arxiv.org/abs/2609.34262) audits passing trajectories for protected-reference access, especially Git history, then patches the access route and reruns agents. This overlaps in reference-exposure auditing and closes off a broad novelty claim.
- [Hardening Agent Benchmarks with Adversarial Hacker-Fixer Loops](https://arxiv.org/abs/2606.08960) studies model-discovered exploits across 1,968 tasks. VARE used scripted candidates on one previously used task; it did not perform model discovery or comparable-scale verifier hardening.
- [Scalable Weak-to-Strong Verifier Hardening for Agent Environments](https://arxiv.org/abs/2610.15924) isolates the verifier and applies an iterative hacker-fixer loop across Terminal Bench, KernelBench, and SETA. This is close prior art for the broader intervention, so the VARE result remains a much narrower case measurement.
- [SWE-Bench Pro V2](https://arxiv.org/abs/2610.24206) is direct precedent for the specific environment-integrity risk: it reports dependency-cache edits and forged module state, regrades patches in a fresh image, and removes `__pycache__` and `.pyc` files before grading. The VARE bytecode observation therefore does not establish a novel defense; it records a small, independently graded transitive-reference case.
- [SWE-Bench Pro Verified](https://arxiv.org/abs/2609.08149) addresses gold-solution/hidden-information reward hacking and task quality. It reinforces that reliable agent evaluation needs both access controls and task validation.

The defensible result is a narrow, preregistered example showing that a transitive dependency exposes equivalent source and bytecode across three pinned images, while source- and bytecode-derived repairs pass because they are genuinely correct. Combined with the separate #1912 false-reward regression and the controlled policy-response measurement, this supplies a small VARE replication with raw artifacts; it does not establish a new general hardening method, broad novelty, agent-discovered exploitation, or a coding-model learning effect.

## Reproduction files

- Frozen protocol and lock: [`protocols/verifier_integrity_transitive_reference_v1.json`](../protocols/verifier_integrity_transitive_reference_v1.json) and [`protocols/verifier_integrity_transitive_reference_v1.lock.json`](../protocols/verifier_integrity_transitive_reference_v1.lock.json).
- Runner and read-only probe: [`scripts/run_verifier_integrity_transitive_reference_v1.py`](../scripts/run_verifier_integrity_transitive_reference_v1.py) and [`benchmarks/verifier_integrity_transitive_reference_v1/access_probe.py`](../benchmarks/verifier_integrity_transitive_reference_v1/access_probe.py).
- Final graded run: [`run4`](../results/verifier_integrity_transitive_reference_v1/run4/); all setup failures remain in [`results/verifier_integrity_transitive_reference_v1/`](../results/verifier_integrity_transitive_reference_v1/).
- Historical #1912 false-reward counterfactuals: [`pilot v1 run 3`](../results/verifier_integrity_pilot_v1/run3/) and the broader context in the [dependency-artifact pilot report](verifier-integrity-dependency-artifacts-v1-report.md).
- Evidence-chain auditor correction: [`PR #123`](https://github.com/mitukx/VARE/pull/123), with the full-chain auditor in [`scripts/audit_verifier_integrity_evidence_v2.py`](../scripts/audit_verifier_integrity_evidence_v2.py).
